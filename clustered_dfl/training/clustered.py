from __future__ import annotations

import heapq

import torch
from torch.utils.data import DataLoader

from clustered_dfl.types import Topology

from .base import ClientState, RunArtifacts, TrainingConfig, log_progress
from .graph import graph_from_edges, metropolis_weights
from .local import train_client_from_vector
from .metrics import evaluate_vector, model_divergence, weighted_global_model
from .time import cluster_update_times
from .vector_utils import average_vectors, initial_model_vector


class ClusteredTrainingRunner:
    def __init__(
        self,
        clients: list[ClientState],
        topology: Topology,
        model_factory,
        test_loader: DataLoader | None,
        config: TrainingConfig,
    ) -> None:
        self.clients = sorted(clients, key=lambda item: item.client_id)
        self.clients_by_id = {client.client_id: client for client in self.clients}
        self.topology = topology
        self.model_factory = model_factory
        self.test_loader = test_loader
        self.config = config
        self.cluster_graph = _cluster_graph(topology)
        self.weights = metropolis_weights(self.cluster_graph)
        self.cluster_sample_counts = [
            sum(self.clients_by_id[cid].metadata.num_samples for cid in cluster)
            for cluster in topology.clusters
        ]
        self.train_model = self.model_factory().to(self.config.device)
        self.eval_model = self.model_factory().to(self.config.device)

    def run(self) -> RunArtifacts:
        init = initial_model_vector(self.model_factory, self.config.seed, self.config.device)
        cluster_models = {cluster_idx: init.clone() for cluster_idx in range(len(self.topology.clusters))}
        cluster_times = cluster_update_times(self.clients_by_id, self.topology, self.config.local_epochs)
        virtual_time = 0.0
        best_accuracy: float | None = None
        metrics: list[dict] = []
        cluster_counter_rows: list[dict] = []
        transmitted = 0.0
        model_size_bytes = float(init.numel() * 4)

        best_accuracy = self._append_metrics(metrics, 0, virtual_time, cluster_models, None, best_accuracy, transmitted)
        counters = {cluster_idx: 0 for cluster_idx in cluster_models}
        cluster_counter_rows.extend(self._cluster_counter_rows(0, virtual_time, counters, cluster_times))
        for step in range(1, self.config.rounds + 1):
            losses = []
            next_cluster_models = {}
            for cluster_idx, members in enumerate(self.topology.clusters):
                start = cluster_models[cluster_idx]
                local_vectors = []
                local_weights = []
                for cid in members:
                    client = self.clients_by_id[cid]
                    vector, loss = train_client_from_vector(
                        self.model_factory,
                        start,
                        client,
                        self.config,
                        model=self.train_model,
                    )
                    local_vectors.append(vector)
                    local_weights.append(float(client.metadata.num_samples))
                    losses.append(loss)
                next_cluster_models[cluster_idx] = average_vectors(local_vectors, local_weights)
                counters[cluster_idx] += 1
                transmitted += 2.0 * model_size_bytes * max(len(members) - 1, 0)

            cluster_models = next_cluster_models
            if self.config.mixing_interval > 0 and step % self.config.mixing_interval == 0:
                cluster_models = self._mix_cluster_models(cluster_models)
                transmitted += 2.0 * model_size_bytes * len(self.topology.leader_edges)
            virtual_time += max(cluster_times) if cluster_times else 0.0

            if step % self.config.eval_interval == 0 or step == self.config.rounds:
                best_accuracy = self._append_metrics(
                    metrics,
                    step,
                    virtual_time,
                    cluster_models,
                    sum(losses) / len(losses) if losses else None,
                    best_accuracy,
                    transmitted,
                )
                cluster_counter_rows.extend(self._cluster_counter_rows(step, virtual_time, counters, cluster_times))

        summary = dict(metrics[-1])
        summary["leaders"] = self.topology.leaders
        summary["leader_edges"] = self.topology.leader_edges
        summary["cluster_update_counts"] = counters
        summary["cluster_sizes"] = [len(cluster) for cluster in self.topology.clusters]
        return RunArtifacts(
            metrics=metrics,
            summary=summary,
            extra_tables={"cluster_counters": cluster_counter_rows},
        )

    def _mix_cluster_models(self, cluster_models: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        reference = {idx: vector.clone() for idx, vector in cluster_models.items()}
        mixed = {}
        for cluster_idx, row in self.weights.items():
            mixed[cluster_idx] = average_vectors([reference[nid] for nid in row], [row[nid] for nid in row])
        return mixed

    def _append_metrics(self, rows, step, virtual_time, cluster_models, train_loss, best_accuracy, transmitted):
        global_model = weighted_global_model(
            [cluster_models[idx] for idx in range(len(self.topology.clusters))],
            self.cluster_sample_counts,
        )
        test_loss, test_accuracy, test_macro_f1 = evaluate_vector(
            self.model_factory,
            global_model,
            self.test_loader,
            self.config.device,
            model=self.eval_model,
        )
        if test_accuracy is not None:
            best_accuracy = test_accuracy if best_accuracy is None else max(best_accuracy, test_accuracy)
        row = {
                "step": step,
                "algorithm": self.config.algorithm,
                "K": len(self.topology.clusters),
                "virtual_time": float(virtual_time),
                "train_loss": train_loss,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "test_macro_f1": test_macro_f1,
                "best_accuracy": best_accuracy,
                "mean_staleness": 0.0,
                "max_staleness": 0.0,
                "leader_edges": str(self.topology.leader_edges),
                "transmitted_bytes_proxy": float(transmitted),
                "model_divergence": model_divergence(list(cluster_models.values()), global_model),
            }
        rows.append(row)
        log_progress(row, self.config.rounds)
        return best_accuracy

    def _cluster_counter_rows(self, step, virtual_time, counters, cluster_times):
        return [
            {
                "step": step,
                "algorithm": self.config.algorithm,
                "K": len(self.topology.clusters),
                "virtual_time": float(virtual_time),
                "cluster_id": cluster_idx,
                "local_update_count": counters.get(cluster_idx, 0),
                "cluster_size": len(self.topology.clusters[cluster_idx]),
                "cluster_update_time": float(cluster_times[cluster_idx]),
            }
            for cluster_idx in range(len(self.topology.clusters))
        ]


class ClusteredBarrierTrainingRunner(ClusteredTrainingRunner):
    def run(self) -> RunArtifacts:
        init = initial_model_vector(self.model_factory, self.config.seed, self.config.device)
        cluster_models = {cluster_idx: init.clone() for cluster_idx in range(len(self.topology.clusters))}
        cluster_times = cluster_update_times(self.clients_by_id, self.topology, self.config.local_epochs)
        virtual_time = 0.0
        best_accuracy: float | None = None
        metrics: list[dict] = []
        cluster_counter_rows: list[dict] = []
        transmitted = 0.0
        model_size_bytes = float(init.numel() * 4)
        sync_comm_time = 0.0
        sync_comm_time_per_mixing = self._sync_comm_time_per_mixing(model_size_bytes)

        best_accuracy = self._append_barrier_metrics(
            metrics,
            0,
            virtual_time,
            cluster_models,
            None,
            best_accuracy,
            transmitted,
            sync_comm_time,
            0.0,
        )
        counters = {cluster_idx: 0 for cluster_idx in cluster_models}
        cluster_counter_rows.extend(self._cluster_counter_rows(0, virtual_time, counters, cluster_times))
        for step in range(1, self.config.rounds + 1):
            losses = []
            next_cluster_models = {}
            for cluster_idx, members in enumerate(self.topology.clusters):
                start = cluster_models[cluster_idx]
                local_vectors = []
                local_weights = []
                for cid in members:
                    client = self.clients_by_id[cid]
                    vector, loss = train_client_from_vector(
                        self.model_factory,
                        start,
                        client,
                        self.config,
                        model=self.train_model,
                    )
                    local_vectors.append(vector)
                    local_weights.append(float(client.metadata.num_samples))
                    losses.append(loss)
                next_cluster_models[cluster_idx] = average_vectors(local_vectors, local_weights)
                counters[cluster_idx] += 1
                transmitted += 2.0 * model_size_bytes * max(len(members) - 1, 0)

            cluster_models = next_cluster_models
            step_sync_comm_time = 0.0
            if self.config.mixing_interval > 0 and step % self.config.mixing_interval == 0:
                cluster_models = self._mix_cluster_models(cluster_models)
                transmitted += 2.0 * model_size_bytes * len(self.topology.leader_edges)
                step_sync_comm_time = sync_comm_time_per_mixing
                sync_comm_time += step_sync_comm_time
            virtual_time += (max(cluster_times) if cluster_times else 0.0) + step_sync_comm_time

            if step % self.config.eval_interval == 0 or step == self.config.rounds:
                best_accuracy = self._append_barrier_metrics(
                    metrics,
                    step,
                    virtual_time,
                    cluster_models,
                    sum(losses) / len(losses) if losses else None,
                    best_accuracy,
                    transmitted,
                    sync_comm_time,
                    step_sync_comm_time,
                )
                cluster_counter_rows.extend(self._cluster_counter_rows(step, virtual_time, counters, cluster_times))

        summary = dict(metrics[-1])
        summary["leaders"] = self.topology.leaders
        summary["leader_edges"] = self.topology.leader_edges
        summary["cluster_update_counts"] = counters
        summary["cluster_sizes"] = [len(cluster) for cluster in self.topology.clusters]
        summary["sync_comm_time_per_mixing"] = sync_comm_time_per_mixing
        return RunArtifacts(
            metrics=metrics,
            summary=summary,
            extra_tables={"cluster_counters": cluster_counter_rows},
        )

    def _sync_comm_time_per_mixing(self, model_size_bytes: float) -> float:
        if self.config.sync_comm_bandwidth_proxy <= 0.0:
            return 0.0
        leader_bytes = 2.0 * model_size_bytes * len(self.topology.leader_edges)
        return leader_bytes / float(self.config.sync_comm_bandwidth_proxy)

    def _append_barrier_metrics(
        self,
        rows,
        step,
        virtual_time,
        cluster_models,
        train_loss,
        best_accuracy,
        transmitted,
        sync_comm_time,
        step_sync_comm_time,
    ):
        global_model = weighted_global_model(
            [cluster_models[idx] for idx in range(len(self.topology.clusters))],
            self.cluster_sample_counts,
        )
        test_loss, test_accuracy, test_macro_f1 = evaluate_vector(
            self.model_factory,
            global_model,
            self.test_loader,
            self.config.device,
            model=self.eval_model,
        )
        if test_accuracy is not None:
            best_accuracy = test_accuracy if best_accuracy is None else max(best_accuracy, test_accuracy)
        row = {
                "step": step,
                "algorithm": self.config.algorithm,
                "K": len(self.topology.clusters),
                "virtual_time": float(virtual_time),
                "train_loss": train_loss,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "test_macro_f1": test_macro_f1,
                "best_accuracy": best_accuracy,
                "mean_staleness": 0.0,
                "max_staleness": 0.0,
                "leader_edges": str(self.topology.leader_edges),
                "transmitted_bytes_proxy": float(transmitted),
                "model_divergence": model_divergence(list(cluster_models.values()), global_model),
                "sync_comm_time_proxy": float(sync_comm_time),
                "step_sync_comm_time_proxy": float(step_sync_comm_time),
                "sync_comm_bandwidth_proxy": float(self.config.sync_comm_bandwidth_proxy),
            }
        rows.append(row)
        log_progress(row, self.config.rounds)
        return best_accuracy


class ClusteredAsyncTrainingRunner(ClusteredTrainingRunner):
    def run(self) -> RunArtifacts:
        init = initial_model_vector(self.model_factory, self.config.seed, self.config.device)
        cluster_models = {cluster_idx: init.clone() for cluster_idx in range(len(self.topology.clusters))}
        counters = {cluster_idx: 0 for cluster_idx in cluster_models}
        cluster_times = cluster_update_times(self.clients_by_id, self.topology, self.config.local_epochs)
        cache = {
            cluster_idx: {neigh: cluster_models[neigh].clone() for neigh in self.cluster_graph[cluster_idx]}
            for cluster_idx in cluster_models
        }
        cache_counters = {
            cluster_idx: {neigh: 0 for neigh in self.cluster_graph[cluster_idx]}
            for cluster_idx in cluster_models
        }
        heap = [(cluster_times[idx], idx) for idx in cluster_models]
        heapq.heapify(heap)
        metrics: list[dict] = []
        cluster_counter_rows: list[dict] = []
        best_accuracy: float | None = None
        transmitted = 0.0
        model_size_bytes = float(init.numel() * 4)
        staleness_values: list[float] = []
        current_time = 0.0

        best_accuracy = self._append_async_metrics(
            metrics,
            0,
            current_time,
            cluster_models,
            None,
            best_accuracy,
            transmitted,
            staleness_values,
        )
        cluster_counter_rows.extend(self._cluster_counter_rows(0, current_time, counters, cluster_times))
        for event in range(1, self.config.events + 1):
            current_time, cluster_idx = heapq.heappop(heap)
            start = cluster_models[cluster_idx]

            cluster_models[cluster_idx], train_loss = self._update_cluster_model(cluster_idx, start)
            counters[cluster_idx] += 1
            transmitted += 2.0 * model_size_bytes * max(len(self.topology.clusters[cluster_idx]) - 1, 0)

            if self.config.mixing_interval > 0 and counters[cluster_idx] % self.config.mixing_interval == 0:
                row = self.weights[cluster_idx]
                vectors = []
                weights = []
                for node, weight in row.items():
                    weights.append(weight)
                    if node == cluster_idx:
                        vectors.append(cluster_models[cluster_idx])
                    else:
                        vectors.append(cache[cluster_idx][node])
                        stale = counters[node] - cache_counters[cluster_idx][node]
                        staleness_values.append(float(stale))
                cluster_models[cluster_idx] = average_vectors(vectors, weights)
                transmitted += 2.0 * model_size_bytes * len(self.cluster_graph[cluster_idx])

            for neigh in self.cluster_graph[cluster_idx]:
                cache[neigh][cluster_idx] = cluster_models[cluster_idx].clone()
                cache_counters[neigh][cluster_idx] = counters[cluster_idx]
            heapq.heappush(heap, (current_time + cluster_times[cluster_idx], cluster_idx))

            if event % self.config.eval_interval == 0 or event == self.config.events:
                best_accuracy = self._append_async_metrics(
                    metrics,
                    event,
                    current_time,
                    cluster_models,
                    train_loss,
                    best_accuracy,
                    transmitted,
                    staleness_values,
                )
                cluster_counter_rows.extend(self._cluster_counter_rows(event, current_time, counters, cluster_times))

        summary = dict(metrics[-1])
        summary["leaders"] = self.topology.leaders
        summary["leader_edges"] = self.topology.leader_edges
        summary["cluster_update_counts"] = counters
        summary["cluster_sizes"] = [len(cluster) for cluster in self.topology.clusters]
        return RunArtifacts(
            metrics=metrics,
            summary=summary,
            extra_tables={"cluster_counters": cluster_counter_rows},
        )

    def _update_cluster_model(self, cluster_idx: int, start: torch.Tensor) -> tuple[torch.Tensor, float | None]:
        local_vectors = []
        local_weights = []
        losses = []
        for cid in self.topology.clusters[cluster_idx]:
            client = self.clients_by_id[cid]
            vector, loss = train_client_from_vector(
                self.model_factory,
                start,
                client,
                self.config,
                model=self.train_model,
            )
            local_vectors.append(vector)
            local_weights.append(float(client.metadata.num_samples))
            losses.append(loss)
        train_loss = sum(losses) / len(losses) if losses else None
        return average_vectors(local_vectors, local_weights), train_loss

    def _append_async_metrics(self, rows, step, virtual_time, cluster_models, train_loss, best_accuracy, transmitted, staleness_values):
        global_model = weighted_global_model(
            [cluster_models[idx] for idx in range(len(self.topology.clusters))],
            self.cluster_sample_counts,
        )
        test_loss, test_accuracy, test_macro_f1 = evaluate_vector(
            self.model_factory,
            global_model,
            self.test_loader,
            self.config.device,
            model=self.eval_model,
        )
        if test_accuracy is not None:
            best_accuracy = test_accuracy if best_accuracy is None else max(best_accuracy, test_accuracy)
        mean_staleness = sum(staleness_values) / len(staleness_values) if staleness_values else 0.0
        max_staleness = max(staleness_values) if staleness_values else 0.0
        row = {
                "step": step,
                "algorithm": self.config.algorithm,
                "K": len(self.topology.clusters),
                "virtual_time": float(virtual_time),
                "train_loss": train_loss,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "test_macro_f1": test_macro_f1,
                "best_accuracy": best_accuracy,
                "mean_staleness": float(mean_staleness),
                "max_staleness": float(max_staleness),
                "leader_edges": str(self.topology.leader_edges),
                "transmitted_bytes_proxy": float(transmitted),
                "model_divergence": model_divergence(list(cluster_models.values()), global_model),
            }
        rows.append(row)
        log_progress(row, self.config.events)
        return best_accuracy


class ClusteredAsyncV1TrainingRunner(ClusteredAsyncTrainingRunner):
    def __init__(
        self,
        clients: list[ClientState],
        topology: Topology,
        model_factory,
        test_loader: DataLoader | None,
        config: TrainingConfig,
    ) -> None:
        super().__init__(clients, topology, model_factory, test_loader, config)
        total_samples = float(sum(self.cluster_sample_counts))
        if total_samples <= 0.0:
            raise ValueError("total cluster sample count must be positive.")
        k = float(len(self.topology.clusters))
        self.cluster_correction_factors = {
            cluster_idx: k * float(sample_count) / total_samples
            for cluster_idx, sample_count in enumerate(self.cluster_sample_counts)
        }

    def _update_cluster_model(self, cluster_idx: int, start: torch.Tensor) -> tuple[torch.Tensor, float | None]:
        local_vectors = []
        local_weights = []
        losses = []
        for cid in self.topology.clusters[cluster_idx]:
            client = self.clients_by_id[cid]
            vector, loss = train_client_from_vector(
                self.model_factory,
                start,
                client,
                self.config,
                model=self.train_model,
            )
            local_vectors.append(vector)
            local_weights.append(float(client.metadata.num_samples))
            losses.append(loss)
        train_loss = sum(losses) / len(losses) if losses else None
        averaged = average_vectors(local_vectors, local_weights)
        correction = self.cluster_correction_factors[cluster_idx]
        corrected_delta = correction * (averaged - start)
        if self.config.cluster_delta_clip_norm > 0.0:
            corrected_delta = _clip_delta(corrected_delta, self.config.cluster_delta_clip_norm)
        return start + corrected_delta, train_loss

    def run(self) -> RunArtifacts:
        artifacts = super().run()
        artifacts.summary["display_algorithm"] = "BLADE-FLv1"
        artifacts.summary["cluster_correction"] = "data_mass"
        artifacts.summary["cluster_delta_clip_norm"] = float(self.config.cluster_delta_clip_norm)
        artifacts.summary["cluster_correction_factors"] = {
            str(cluster_idx): float(value)
            for cluster_idx, value in self.cluster_correction_factors.items()
        }
        return artifacts


def _cluster_graph(topology: Topology):
    leader_to_cluster = {leader: idx for idx, leader in enumerate(topology.leaders)}
    edges = []
    for left, right in topology.leader_edges:
        if left in leader_to_cluster and right in leader_to_cluster:
            edges.append((leader_to_cluster[left], leader_to_cluster[right]))
    return graph_from_edges(list(range(len(topology.clusters))), edges)


class ClusteredAsyncClipProxTrainingRunner(ClusteredAsyncTrainingRunner):
    def __init__(
        self,
        clients: list[ClientState],
        topology: Topology,
        model_factory,
        test_loader: DataLoader | None,
        config: TrainingConfig,
    ) -> None:
        super().__init__(clients, topology, model_factory, test_loader, config)
        if self.config.clipprox_clip_norm <= 0.0:
            raise ValueError("clipprox_clip_norm must be positive.")
        if self.config.clipprox_mu < 0.0:
            raise ValueError("clipprox_mu must be non-negative.")

    def _update_cluster_model(self, cluster_idx: int, start: torch.Tensor) -> tuple[torch.Tensor, float | None]:
        clipped_deltas = []
        local_weights = []
        losses = []
        for cid in self.topology.clusters[cluster_idx]:
            client = self.clients_by_id[cid]
            vector, loss = train_client_from_vector(
                self.model_factory,
                start,
                client,
                self.config,
                prox_reference_vector=start,
                prox_mu=self.config.clipprox_mu,
                model=self.train_model,
            )
            clipped_deltas.append(_clip_delta(vector - start, self.config.clipprox_clip_norm))
            local_weights.append(float(client.metadata.num_samples))
            losses.append(loss)
        train_loss = sum(losses) / len(losses) if losses else None
        return start + average_vectors(clipped_deltas, local_weights), train_loss


def _clip_delta(delta: torch.Tensor, clip_norm: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(delta)
    if float(norm) <= clip_norm:
        return delta
    return delta * (clip_norm / (norm + 1.0e-12))
