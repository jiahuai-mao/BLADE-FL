from __future__ import annotations

import argparse
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from iotlab_dfl.model import evaluate, weighted_average
from iotlab_dfl.protocol import rpc
from iotlab_dfl.storage import append_jsonl, read_json, write_csv, write_json
from iotlab_dfl.topology import build_cluster_topology, metropolis_weights, random_connected_graph, sparse_regular_graph


ALGORITHMS = ("bladefl", "sparse_dpsgd", "adpsgd", "mdfeel")


class Controller:
    def __init__(self, args):
        self.args = args
        self.bundle = Path(args.bundle_dir).resolve()
        self.manifest = read_json(self.bundle / "manifest.json")
        self.config = read_json(args.config)
        self.nodes = sorted(self.config["nodes"], key=lambda item: int(item["client_id"]))
        self.nodes_by_id = {int(item["client_id"]): item for item in self.nodes}
        timestamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        self.run_id = args.run_id or "%s_seed%d_%s" % (args.algorithm, args.seed, timestamp)
        self.output = Path(args.output_dir).resolve() / self.run_id
        self.output.mkdir(parents=True, exist_ok=False)
        self.controller_log = self.output / "controller_messages.jsonl"
        self.evaluations = []
        self.run_started = None
        self.eval_pause_sec = 0.0
        self.algorithm_setup_bytes = 0
        self.counter_lock = threading.Lock()
        self.topology = None
        self.initial_model = np.load(self.bundle / "initial_model.npy").astype(np.float32)
        with np.load(self.bundle / "data" / "test.npz") as payload:
            self.test_x = payload["x"].astype(np.float32, copy=False)
            self.test_y = payload["y"].astype(np.int64, copy=False)

    def execute(self):
        health = self.call_all({"op": "health"}, category="instrumentation")
        self._validate_health(health)
        self.run_started = time.time()
        run = self._run_config()
        self.call_all({"op": "configure", "run": run}, category="orchestration")
        topology_started = time.time()
        self.topology = self._build_topology()
        topology_elapsed = time.time() - topology_started
        if self.topology is not None:
            self.topology["controller_topology_wall_time_sec"] = topology_elapsed
            write_json(self.output / "topology.json", self.topology)
        write_json(self.output / "run.json", {**run, "nodes": self.nodes, "topology": self.topology})

        self.evaluate_checkpoint(step=0, equivalent_round=0.0)
        if self.args.algorithm == "sparse_dpsgd":
            self._run_sparse_dpsgd()
        elif self.args.algorithm == "adpsgd":
            self._run_adpsgd()
        else:
            self._run_clustered()
        self.call_all({"op": "stop"}, category="orchestration")
        self._wait_for_workers()
        statuses = self.status_all()
        completed = sum(int(item["completed_updates"]) for item in statuses)
        self.evaluate_checkpoint(step=completed, equivalent_round=completed / float(len(self.nodes)), final=True)
        self._flush_node_logs()
        summary = self._summary(statuses)
        write_json(self.output / "summary.json", summary)
        write_csv(self.output / "evaluations.csv", self.evaluations)
        print(json.dumps(summary, indent=2, sort_keys=True))

    def _run_config(self):
        return {
            "run_id": self.run_id,
            "algorithm": self.args.algorithm,
            "seed": self.args.seed,
            "dataset": "hhar",
            "num_physical_nodes": len(self.nodes),
            "num_logical_clients": len(self.nodes),
            "batch_size": int(self.config["batch_size"]),
            "local_epochs": int(self.config["local_epochs"]),
            "learning_rate": float(self.config["learning_rate"]),
            "min_learning_rate": float(self.config.get("min_learning_rate", self.config["learning_rate"])),
            "lr_schedule": str(self.config.get("lr_schedule", "constant")),
            "momentum": float(self.config["momentum"]),
            "weight_decay": float(self.config["weight_decay"]),
            "rounds": int(self.args.rounds or self.config["rounds"]),
            "target_client_updates": int((self.args.rounds or self.config["rounds"]) * len(self.nodes)),
            "eval_every_equivalent_rounds": int(self.config["eval_every_equivalent_rounds"]),
            "mixing_interval": int(self.config["mixing_interval"]),
            "sparse_degree": int(self.config["sparse_degree"]),
            "random_edge_probability": float(self.config["random_edge_probability"]),
            "socket_timeout_sec": float(self.config.get("socket_timeout_sec", 120.0)),
            "model_parameter_count": int(self.manifest["parameter_count"]),
            "model_bytes": int(self.manifest["model_bytes"]),
            "communication_definition": "sum of request and response application-layer bytes for peer training RPCs and algorithm-specific topology setup; evaluation, monitoring, and generic orchestration excluded",
            "time_definition": "elapsed from configure through training, including topology setup; synchronous evaluation pauses excluded",
            "node_log_dir": "/tmp/iotlab_dfl",
            "run_start_epoch": float(self.run_started),
        }

    def _build_topology(self):
        ids = [int(item["client_id"]) for item in self.nodes]
        if self.args.algorithm == "sparse_dpsgd":
            graph = sparse_regular_graph(ids, int(self.config["sparse_degree"]))
            return {"method": "sparse_dpsgd", "degree": int(self.config["sparse_degree"]), "graph": json_graph(graph)}
        if self.args.algorithm == "adpsgd":
            graph = random_connected_graph(ids, float(self.config["random_edge_probability"]), self.args.seed)
            return {"method": "adpsgd", "edge_probability": float(self.config["random_edge_probability"]), "graph": json_graph(graph)}

        profiles = self.call_all({"op": "profile", "profile_seed": self.args.seed}, category="algorithm_setup")
        profile_by_id = {int(item["client_id"]): item for item in profiles}
        clients = []
        for client in self.manifest["clients"]:
            client_id = int(client["client_id"])
            clients.append(
                {
                    "client_id": client_id,
                    "num_samples": int(client["num_samples"]),
                    "label_counts": list(profile_by_id[client_id]["label_counts"]),
                    "profile_train_sec": float(profile_by_id[client_id]["profile_train_sec"]),
                }
            )
        k_max = self.config.get("k_max", "sqrt")
        if str(k_max).lower() == "sqrt":
            k_max = int(math.floor(math.sqrt(len(clients))))
        else:
            k_max = int(k_max)
        topology = build_cluster_topology(
            clients,
            self.args.algorithm,
            k_min=int(self.config.get("k_min", 2)),
            k_max=k_max,
            min_cluster_size=int(self.config.get("min_clients_per_cluster", 2)),
            max_swaps=int(self.config.get("max_swaps", 10)),
            mixing_interval=int(self.config["mixing_interval"]),
        )
        topology["clients"] = clients
        return topology

    def _run_sparse_dpsgd(self):
        graph = {int(key): set(value) for key, value in self.topology["graph"].items()}
        weights = metropolis_weights(graph)
        self.call_per_node(
            {
                client_id: {"op": "set_topology", "topology": {"neighbors": sorted(graph[client_id]), "weights": weights[client_id]}}
                for client_id in graph
            },
            category="algorithm_setup",
        )
        rounds = int(self.args.rounds or self.config["rounds"])
        eval_every = int(self.config["eval_every_equivalent_rounds"])
        for round_id in range(1, rounds + 1):
            push_commands = {}
            for client_id in graph:
                push_commands[client_id] = {
                    "op": "push_current",
                    "peers": [self.nodes_by_id[peer] for peer in sorted(graph[client_id])],
                    "message_type": "sparse_dpsgd_model",
                }
            self.call_per_node(push_commands, category="orchestration")
            train_commands = {
                client_id: {"op": "mix_train", "weights": weights[client_id], "event": round_id}
                for client_id in graph
            }
            self.call_per_node(train_commands, category="orchestration")
            if round_id % eval_every == 0 or round_id == rounds:
                pause_started = time.time()
                self.evaluate_checkpoint(step=round_id * len(self.nodes), equivalent_round=float(round_id))
                self.eval_pause_sec += time.time() - pause_started

    def _run_adpsgd(self):
        graph = {int(key): set(value) for key, value in self.topology["graph"].items()}
        weights = metropolis_weights(graph)
        commands = {}
        rounds = int(self.args.rounds or self.config["rounds"])
        for client_id in graph:
            commands[client_id] = {
                "op": "start_adpsgd",
                "peers": [self.nodes_by_id[peer] for peer in sorted(graph[client_id])],
                "weights": weights[client_id],
                "max_update_skew": self.args.max_update_skew,
                "max_local_updates": rounds,
            }
        self.call_per_node(commands, category="algorithm_setup")
        self._monitor_async()

    def _run_clustered(self):
        leader_neighbors = {int(leader): set() for leader in self.topology["leaders"]}
        for left, right in self.topology["leader_edges"]:
            leader_neighbors[int(left)].add(int(right))
            leader_neighbors[int(right)].add(int(left))
        commands = {}
        rounds = int(self.args.rounds or self.config["rounds"])
        for cluster_id, cluster in enumerate(self.topology["clusters"]):
            leader = int(self.topology["leaders"][cluster_id])
            commands[leader] = {
                "op": "start_cluster",
                "cluster_id": cluster_id,
                "members": [self.nodes_by_id[int(client_id)] for client_id in cluster],
                "leader_peers": [self.nodes_by_id[item] for item in sorted(leader_neighbors[leader])],
                "mixing_interval": int(self.config["mixing_interval"]),
                "max_cluster_events": rounds,
            }
        self.call_per_node(commands, category="algorithm_setup")
        self._monitor_async()

    def _monitor_async(self):
        target = int((self.args.rounds or self.config["rounds"]) * len(self.nodes))
        interval = int(self.config["eval_every_equivalent_rounds"]) * len(self.nodes)
        next_eval = interval
        while True:
            statuses = self.status_all()
            completed = sum(int(item["completed_updates"]) for item in statuses)
            if completed >= next_eval:
                self.evaluate_checkpoint(step=completed, equivalent_round=completed / float(len(self.nodes)))
                next_eval += interval
            if completed >= target:
                break
            time.sleep(float(self.config.get("poll_interval_sec", 1.0)))

    def evaluate_checkpoint(self, step, equivalent_round, final=False):
        if self.topology and self.topology.get("clusters"):
            selected = [int(item) for item in self.topology["leaders"]]
            weights = [sum(int(self.manifest["clients"][cid]["num_samples"]) for cid in cluster) for cluster in self.topology["clusters"]]
        else:
            selected = [int(item["client_id"]) for item in self.nodes]
            weights = [int(self.manifest["clients"][cid]["num_samples"]) for cid in selected]
        responses = self.call_per_node(
            {client_id: {"op": "snapshot"} for client_id in selected},
            category="evaluation",
            include_vectors=True,
        )
        vectors = [responses[client_id][1] for client_id in selected]
        global_model = weighted_average(vectors, weights)
        test_loss, accuracy, macro_f1 = evaluate(
            global_model,
            self.test_x,
            self.test_y,
            int(self.manifest["input_dim"]),
            self.manifest.get("hidden_dims", [int(self.manifest["hidden_dim"])]),
            int(self.manifest["num_classes"]),
        )
        statuses = self.status_all()
        peer_training_bytes = sum(int(item.get("training_bytes", 0)) for item in statuses)
        with self.counter_lock:
            setup_bytes = self.algorithm_setup_bytes
        training_bytes = peer_training_bytes + setup_bytes
        divergence = float(np.average([float(np.linalg.norm(vector - global_model)) for vector in vectors], weights=weights))
        elapsed = time.time() - self.run_started - self.eval_pause_sec
        row = {
            "step": int(step),
            "equivalent_round": float(equivalent_round),
            "wall_clock_sec": float(max(elapsed, 0.0)),
            "test_loss": test_loss,
            "test_accuracy": accuracy,
            "test_macro_f1": macro_f1,
            "model_divergence": divergence,
            "training_transmitted_bytes": int(training_bytes),
            "peer_training_bytes": int(peer_training_bytes),
            "algorithm_setup_bytes": int(setup_bytes),
            "completed_client_updates": int(sum(int(item["completed_updates"]) for item in statuses)),
            "final": bool(final),
        }
        self.evaluations.append(row)
        write_csv(self.output / "evaluations.csv", self.evaluations)
        np.savez_compressed(self.output / ("model_step_%08d.npz" % int(step)), global_model=global_model)
        print("[eval] algorithm=%s updates=%d time=%.2fs bytes=%d accuracy=%.4f" % (self.args.algorithm, row["completed_client_updates"], row["wall_clock_sec"], training_bytes, accuracy), flush=True)

    def status_all(self):
        return self.call_all({"op": "status"}, category="monitoring")

    def call_all(self, header, category, include_vectors=False):
        commands = {int(node["client_id"]): dict(header) for node in self.nodes}
        result = self.call_per_node(commands, category, include_vectors=include_vectors)
        if include_vectors:
            return result
        return [result[key][0] for key in sorted(result)]

    def call_per_node(self, commands, category, include_vectors=False):
        result = {}
        with ThreadPoolExecutor(max_workers=max(1, len(commands))) as executor:
            futures = {
                executor.submit(self._call_node, self.nodes_by_id[int(client_id)], command, category): int(client_id)
                for client_id, command in commands.items()
            }
            for future in as_completed(futures):
                client_id = futures[future]
                result[client_id] = future.result()
        return result

    def _call_node(self, node, header, category):
        response, vector, stats = rpc(
            node["hostname"],
            node["port"],
            header,
            timeout=float(self.config.get("socket_timeout_sec", 120.0)),
        )
        append_jsonl(
            self.controller_log,
            {
                "timestamp": time.time(),
                "client_id": int(node["client_id"]),
                "operation": header.get("op"),
                "category": category,
                "request_bytes": stats["request_bytes"],
                "response_bytes": stats["response_bytes"],
                "transmitted_bytes": stats["total_bytes"],
                "elapsed_sec": stats["elapsed_sec"],
            },
        )
        if category == "algorithm_setup":
            with self.counter_lock:
                self.algorithm_setup_bytes += int(stats["total_bytes"])
        return response, vector, stats

    def _wait_for_workers(self):
        deadline = time.time() + 30.0
        while time.time() < deadline:
            statuses = self.status_all()
            if not any(item["worker_alive"] for item in statuses):
                return
            time.sleep(0.2)

    def _flush_node_logs(self):
        node_output = self.args.node_output_dir.rstrip("/") + "/" + self.run_id + "/node_logs"
        self.call_all({"op": "flush_logs", "output_dir": node_output}, category="instrumentation")

    def _summary(self, statuses):
        final = self.evaluations[-1]
        return {
            "run_id": self.run_id,
            "algorithm": self.args.algorithm,
            "seed": self.args.seed,
            "num_nodes": len(self.nodes),
            "completed_client_updates": sum(int(item["completed_updates"]) for item in statuses),
            "final_accuracy": final["test_accuracy"],
            "final_macro_f1": final["test_macro_f1"],
            "wall_clock_sec": final["wall_clock_sec"],
            "training_transmitted_bytes": final["training_transmitted_bytes"],
            "peer_training_bytes": final["peer_training_bytes"],
            "algorithm_setup_bytes": final["algorithm_setup_bytes"],
            "topology_construction_sec": 0.0 if self.topology is None else self.topology.get("controller_topology_wall_time_sec", 0.0),
            "selected_k": None if self.topology is None else self.topology.get("K"),
            "output_dir": str(self.output),
            "run_end_epoch": time.time(),
        }

    def _validate_health(self, health):
        ids = sorted(int(item["client_id"]) for item in health)
        expected = sorted(int(item["client_id"]) for item in self.nodes)
        if ids != expected:
            raise RuntimeError("agent client ids do not match configuration")
        parameter_counts = sorted(set(int(item["model_parameters"]) for item in health))
        expected_parameters = int(self.initial_model.size)
        if parameter_counts != [expected_parameters]:
            raise RuntimeError(
                "agent model sizes %s do not match controller model size %d"
                % (parameter_counts, expected_parameters)
            )
        versions = sorted(set(item["numpy"] for item in health))
        feature_modes = sorted(set(str(item["feature_mode"]) for item in health))
        print(
            "[controller] ready nodes=%d numpy=%s model_parameters=%d features=%s"
            % (len(health), ",".join(versions), expected_parameters, ",".join(feature_modes)),
            flush=True,
        )


def json_graph(graph):
    return {str(key): sorted(int(item) for item in value) for key, value in graph.items()}


def parse_args():
    frontend_root = Path.home() / "shared" / "iotlab_dfl"
    parser = argparse.ArgumentParser(description="Run a physical FIT IoT-LAB HHAR validation experiment.")
    parser.add_argument("--algorithm", choices=ALGORITHMS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--bundle-dir", default=str(frontend_root / "bundle"))
    parser.add_argument("--config", default=str(frontend_root / "bundle" / "iotlab_config.json"))
    parser.add_argument("--output-dir", default=str(frontend_root / "results"))
    parser.add_argument("--node-output-dir", default="/home/root/shared/iotlab_dfl/results")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-update-skew", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    Controller(parse_args()).execute()
