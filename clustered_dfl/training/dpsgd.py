from __future__ import annotations

import heapq
from collections.abc import Mapping

import torch
from torch.utils.data import DataLoader

from .base import ClientState, RunArtifacts, TrainingConfig
from .graph import graph_edges, metropolis_weights
from .local import train_client_from_vector
from .metrics import evaluate_vector, model_divergence, weighted_global_model
from .time import client_train_time
from .vector_utils import average_vectors, initial_model_vector


class DPSGDRunner:
    def __init__(
        self,
        clients: list[ClientState],
        graph: Mapping[int, set[int]],
        model_factory,
        test_loader: DataLoader | None,
        config: TrainingConfig,
    ) -> None:
        self.clients = sorted(clients, key=lambda item: item.client_id)
        self.clients_by_id = {client.client_id: client for client in self.clients}
        self.graph = {int(node): {int(neigh) for neigh in neighbors} for node, neighbors in graph.items()}
        self.weights = metropolis_weights(self.graph)
        self.model_factory = model_factory
        self.test_loader = test_loader
        self.config = config

    def run(self) -> RunArtifacts:
        init = initial_model_vector(self.model_factory, self.config.seed, self.config.device)
        client_models = {client.client_id: init.clone() for client in self.clients}
        virtual_time = 0.0
        best_accuracy: float | None = None
        metrics: list[dict] = []
        transmitted = 0.0
        model_size_bytes = float(init.numel() * 4)
        edges = graph_edges(self.graph)

        best_accuracy = self._append_metrics(metrics, 0, virtual_time, client_models, None, best_accuracy, transmitted)
        for step in range(1, self.config.rounds + 1):
            reference = {cid: vector.clone() for cid, vector in client_models.items()}
            mixed = {}
            for cid in reference:
                row = self.weights[cid]
                mixed[cid] = average_vectors([reference[nid] for nid in row], [row[nid] for nid in row])

            losses = []
            for client in self.clients:
                new_vector, loss = train_client_from_vector(self.model_factory, mixed[client.client_id], client, self.config)
                client_models[client.client_id] = new_vector
                losses.append(loss)
            virtual_time += max(client_train_time(client, self.config.local_epochs) for client in self.clients)
            transmitted += 2.0 * model_size_bytes * len(edges)

            if step % self.config.eval_interval == 0 or step == self.config.rounds:
                best_accuracy = self._append_metrics(
                    metrics,
                    step,
                    virtual_time,
                    client_models,
                    sum(losses) / len(losses) if losses else None,
                    best_accuracy,
                    transmitted,
                )

        summary = dict(metrics[-1])
        summary["graph_edges"] = edges
        summary["num_clients"] = len(self.clients)
        return RunArtifacts(metrics=metrics, summary=summary)

    def _append_metrics(self, rows, step, virtual_time, client_models, train_loss, best_accuracy, transmitted):
        global_model = weighted_global_model(
            [client_models[client.client_id] for client in self.clients],
            [client.metadata.num_samples for client in self.clients],
        )
        test_loss, test_accuracy = evaluate_vector(self.model_factory, global_model, self.test_loader, self.config.device)
        if test_accuracy is not None:
            best_accuracy = test_accuracy if best_accuracy is None else max(best_accuracy, test_accuracy)
        rows.append(
            {
                "step": step,
                "algorithm": "dpsgd",
                "K": "",
                "virtual_time": float(virtual_time),
                "train_loss": train_loss,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "best_accuracy": best_accuracy,
                "mean_staleness": 0.0,
                "max_staleness": 0.0,
                "leader_edges": "",
                "transmitted_bytes_proxy": float(transmitted),
                "model_divergence": model_divergence(list(client_models.values()), global_model),
            }
        )
        return best_accuracy


class ADPSGDRunner:
    def __init__(
        self,
        clients: list[ClientState],
        graph: Mapping[int, set[int]],
        model_factory,
        test_loader: DataLoader | None,
        config: TrainingConfig,
    ) -> None:
        self.clients = sorted(clients, key=lambda item: item.client_id)
        self.clients_by_id = {client.client_id: client for client in self.clients}
        self.graph = {int(node): {int(neigh) for neigh in neighbors} for node, neighbors in graph.items()}
        self.weights = metropolis_weights(self.graph)
        self.model_factory = model_factory
        self.test_loader = test_loader
        self.config = config

    def run(self) -> RunArtifacts:
        init = initial_model_vector(self.model_factory, self.config.seed, self.config.device)
        client_models = {client.client_id: init.clone() for client in self.clients}
        counters = {client.client_id: 0 for client in self.clients}
        cache = {
            cid: {neigh: client_models[neigh].clone() for neigh in self.graph[cid]}
            for cid in client_models
        }
        cache_counters = {
            cid: {neigh: 0 for neigh in self.graph[cid]}
            for cid in client_models
        }
        heap = [
            (client_train_time(client, self.config.local_epochs), client.client_id)
            for client in self.clients
        ]
        heapq.heapify(heap)
        metrics: list[dict] = []
        best_accuracy: float | None = None
        transmitted = 0.0
        model_size_bytes = float(init.numel() * 4)
        staleness_values: list[float] = []
        current_time = 0.0

        best_accuracy = self._append_metrics(
            metrics,
            0,
            current_time,
            client_models,
            None,
            best_accuracy,
            transmitted,
            staleness_values,
        )
        for event in range(1, self.config.events + 1):
            event_time, cid = self._pop_next_event(heap, counters)
            current_time = max(current_time, event_time)
            row = self.weights[cid]
            vectors = []
            weights = []
            for node, weight in row.items():
                weights.append(weight)
                if node == cid:
                    vectors.append(client_models[cid])
                else:
                    vectors.append(cache[cid][node])
                    stale = counters[node] - cache_counters[cid][node]
                    staleness_values.append(float(stale))
            mixed = average_vectors(vectors, weights)
            client = self.clients_by_id[cid]
            new_vector, train_loss = train_client_from_vector(self.model_factory, mixed, client, self.config)
            client_models[cid] = new_vector
            counters[cid] += 1
            for neigh in self.graph[cid]:
                cache[neigh][cid] = new_vector.clone()
                cache_counters[neigh][cid] = counters[cid]
            transmitted += 2.0 * model_size_bytes * len(self.graph[cid])
            heapq.heappush(heap, (current_time + client_train_time(client, self.config.local_epochs), cid))

            if event % self.config.eval_interval == 0 or event == self.config.events:
                best_accuracy = self._append_metrics(
                    metrics,
                    event,
                    current_time,
                    client_models,
                    train_loss,
                    best_accuracy,
                    transmitted,
                    staleness_values,
                )

        summary = dict(metrics[-1])
        summary["client_update_counts"] = counters
        summary["graph_edges"] = graph_edges(self.graph)
        summary["max_async_update_skew"] = self.config.max_async_update_skew
        return RunArtifacts(metrics=metrics, summary=summary)

    def _pop_next_event(self, heap, counters):
        max_skew = self.config.max_async_update_skew
        if max_skew < 0:
            return heapq.heappop(heap)
        max_skew = max(1, max_skew)

        deferred = []
        while heap:
            event_time, cid = heapq.heappop(heap)
            min_count = min(counters.values())
            if counters[cid] < min_count + max_skew:
                for item in deferred:
                    heapq.heappush(heap, item)
                return event_time, cid
            deferred.append((event_time, cid))

        for item in deferred:
            heapq.heappush(heap, item)
        return heapq.heappop(heap)

    def _append_metrics(self, rows, step, virtual_time, client_models, train_loss, best_accuracy, transmitted, staleness_values):
        global_model = weighted_global_model(
            [client_models[client.client_id] for client in self.clients],
            [client.metadata.num_samples for client in self.clients],
        )
        test_loss, test_accuracy = evaluate_vector(self.model_factory, global_model, self.test_loader, self.config.device)
        if test_accuracy is not None:
            best_accuracy = test_accuracy if best_accuracy is None else max(best_accuracy, test_accuracy)
        mean_staleness = sum(staleness_values) / len(staleness_values) if staleness_values else 0.0
        max_staleness = max(staleness_values) if staleness_values else 0.0
        rows.append(
            {
                "step": step,
                "algorithm": "adpsgd",
                "K": "",
                "virtual_time": float(virtual_time),
                "train_loss": train_loss,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "best_accuracy": best_accuracy,
                "mean_staleness": float(mean_staleness),
                "max_staleness": float(max_staleness),
                "leader_edges": "",
                "transmitted_bytes_proxy": float(transmitted),
                "model_divergence": model_divergence(list(client_models.values()), global_model),
            }
        )
        return best_accuracy
