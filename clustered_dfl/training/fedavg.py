from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from .base import ClientState, RunArtifacts, TrainingConfig, log_progress
from .local import train_client_from_vector
from .metrics import evaluate_vector
from .time import client_train_time
from .vector_utils import average_vectors, initial_model_vector


class FedAvgRunner:
    def __init__(
        self,
        clients: list[ClientState],
        model_factory,
        test_loader: DataLoader | None,
        config: TrainingConfig,
    ) -> None:
        self.clients = sorted(clients, key=lambda item: item.client_id)
        self.model_factory = model_factory
        self.test_loader = test_loader
        self.config = config
        self.train_model = self.model_factory().to(self.config.device)
        self.eval_model = self.model_factory().to(self.config.device)

    def run(self) -> RunArtifacts:
        global_model = initial_model_vector(self.model_factory, self.config.seed, self.config.device)
        virtual_time = 0.0
        best_accuracy: float | None = None
        metrics: list[dict] = []
        transmitted = 0.0
        model_size_bytes = float(global_model.numel() * 4)

        best_accuracy = self._append_metrics(metrics, 0, virtual_time, global_model, None, best_accuracy, transmitted)
        for step in range(1, self.config.rounds + 1):
            local_vectors = []
            local_weights = []
            losses = []
            for client in self.clients:
                vector, loss = train_client_from_vector(
                    self.model_factory,
                    global_model,
                    client,
                    self.config,
                    model=self.train_model,
                )
                local_vectors.append(vector)
                local_weights.append(float(client.metadata.num_samples))
                losses.append(loss)
            global_model = average_vectors(local_vectors, local_weights)
            virtual_time += max(client_train_time(client, self.config.local_epochs) for client in self.clients)
            transmitted += 2.0 * model_size_bytes * len(self.clients)

            if step % self.config.eval_interval == 0 or step == self.config.rounds:
                best_accuracy = self._append_metrics(
                    metrics,
                    step,
                    virtual_time,
                    global_model,
                    sum(losses) / len(losses) if losses else None,
                    best_accuracy,
                    transmitted,
                )

        summary = dict(metrics[-1])
        summary["num_clients"] = len(self.clients)
        return RunArtifacts(metrics=metrics, summary=summary)

    def _append_metrics(self, rows, step, virtual_time, global_model, train_loss, best_accuracy, transmitted):
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
                "algorithm": "fedavg",
                "K": "",
                "virtual_time": float(virtual_time),
                "train_loss": train_loss,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "test_macro_f1": test_macro_f1,
                "best_accuracy": best_accuracy,
                "mean_staleness": 0.0,
                "max_staleness": 0.0,
                "leader_edges": "",
                "transmitted_bytes_proxy": float(transmitted),
                "model_divergence": 0.0,
            }
        rows.append(row)
        log_progress(row, self.config.rounds)
        return best_accuracy
