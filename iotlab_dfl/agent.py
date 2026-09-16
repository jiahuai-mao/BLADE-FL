from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import socketserver
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from iotlab_dfl.model import cosine_learning_rate, train_local, weighted_average
from iotlab_dfl.protocol import receive_packet, rpc, send_packet
from iotlab_dfl.storage import append_jsonl, read_json


class AgentState:
    def __init__(self, bundle_dir, config_path, client_id=None):
        self.bundle_dir = Path(bundle_dir).resolve()
        self.manifest = read_json(self.bundle_dir / "manifest.json")
        self.iotlab_config = read_json(config_path)
        self.hostname = socket.gethostname().split(".")[0]
        self.node = self._resolve_node(client_id)
        self.client_id = int(self.node["client_id"])
        client = self.manifest["clients"][self.client_id]
        with np.load(self.bundle_dir / client["file"]) as payload:
            self.x = payload["x"].astype(np.float32, copy=False)
            self.y = payload["y"].astype(np.int64, copy=False)
        self.num_samples = int(len(self.y))
        self.model = np.load(self.bundle_dir / "initial_model.npy").astype(np.float32)
        self.model_version = 0
        self.completed_updates = 0
        self.cluster_events = 0
        self.training_bytes = 0
        self.peer_cache = {}
        self.peer_counters = {}
        self.run = {}
        self.run_id = "unconfigured"
        self.stop_event = threading.Event()
        self.worker = None
        self.lock = threading.RLock()
        self.training_lock = threading.Lock()
        self.local_log = None

    def _resolve_node(self, explicit_client_id):
        if explicit_client_id is not None:
            return next(item for item in self.iotlab_config["nodes"] if int(item["client_id"]) == explicit_client_id)
        for item in self.iotlab_config["nodes"]:
            if str(item["hostname"]).split(".")[0] == self.hostname:
                return item
        if self.hostname.startswith("node-a8-"):
            try:
                node_id = int(self.hostname.rsplit("-", 1)[1])
            except ValueError:
                node_id = None
            for item in self.iotlab_config["nodes"]:
                if node_id is not None and int(item.get("node_id", -1)) == node_id:
                    return item
        raise RuntimeError("hostname %s is absent from IoT-LAB configuration" % self.hostname)

    def handle(self, header, vector):
        operation = header.get("op")
        if operation == "health":
            return self._health(), None
        if operation == "configure":
            return self._configure(header["run"]), None
        if operation == "set_model":
            return self._set_model(vector, header), None
        if operation == "set_topology":
            return {"ok": True, "client_id": self.client_id, "topology": header.get("topology")}, None
        if operation == "profile":
            return self._profile(header), None
        if operation == "train_from":
            return self._train_from(vector, header)
        if operation == "receive_peer":
            return self._receive_peer(vector, header), None
        if operation == "push_current":
            return self._push_current(header), None
        if operation == "mix_train":
            return self._mix_train(header), None
        if operation == "start_adpsgd":
            return self._start_worker(self._adpsgd_loop, header), None
        if operation == "start_cluster":
            return self._start_worker(self._cluster_loop, header), None
        if operation == "status":
            return self._status(), None
        if operation == "snapshot":
            with self.lock:
                return self._status(), self.model.copy()
        if operation == "stop":
            self.stop_event.set()
            return self._status(), None
        if operation == "flush_logs":
            return self._flush_logs(header["output_dir"]), None
        raise ValueError("unknown operation: %s" % operation)

    def _health(self):
        return {
            "ok": True,
            "hostname": self.hostname,
            "client_id": self.client_id,
            "python": tuple(__import__("sys").version_info[:3]),
            "numpy": np.__version__,
            "num_samples": self.num_samples,
            "model_parameters": int(self.model.size),
            "feature_mode": self.manifest.get("feature_mode", "legacy12"),
        }

    def _configure(self, run):
        self.stop_event.set()
        if self.worker is not None and self.worker.is_alive():
            self.worker.join(timeout=5.0)
        with self.lock:
            self.run = dict(run)
            self.run_id = str(run["run_id"])
            self.model = np.load(self.bundle_dir / "initial_model.npy").astype(np.float32)
            self.model_version = 0
            self.completed_updates = 0
            self.cluster_events = 0
            self.training_bytes = 0
            self.peer_cache = {}
            self.peer_counters = {}
            self.stop_event = threading.Event()
            log_dir = Path(run.get("node_log_dir", "/tmp/iotlab_dfl")) / self.run_id
            log_dir.mkdir(parents=True, exist_ok=True)
            self.local_log = log_dir / ("%s_client%02d.jsonl" % (self.hostname, self.client_id))
            if self.local_log.exists():
                self.local_log.unlink()
            self._log({"kind": "lifecycle", "event": "configured", "timestamp": time.time()})
        return self._status()

    def _set_model(self, vector, header):
        self._require_vector(vector)
        with self.lock:
            self.model = vector.astype(np.float32, copy=True)
            self.model_version = int(header.get("model_version", 0))
        return self._status()

    def _profile(self, header):
        with self.lock:
            start = self.model.copy()
        started = time.time()
        result, loss, batches = self._local_train(start, int(header.get("profile_seed", self.client_id)), 0)
        ended = time.time()
        self._log_interval("LOCAL_TRAIN", started, ended, {"profile": True})
        return {
            "ok": True,
            "client_id": self.client_id,
            "num_samples": self.num_samples,
            "label_counts": np.bincount(
                self.y.astype(np.intp, copy=False), minlength=int(self.manifest["num_classes"])
            ).astype(int).tolist(),
            "profile_train_sec": ended - started,
            "profile_loss": loss,
            "num_batches": batches,
            "model_norm": float(np.linalg.norm(result)),
        }

    def _train_from(self, vector, header):
        self._require_vector(vector)
        event = int(header.get("event", 0))
        seed = self._update_seed(event)
        started = time.time()
        trained, loss, batches = self._local_train(vector, seed, event)
        ended = time.time()
        with self.lock:
            self.completed_updates += 1
            update = self.completed_updates
        self._log_update(started, ended, loss, batches, event, header.get("cluster_id"), update)
        return {
            "ok": True,
            "client_id": self.client_id,
            "num_samples": self.num_samples,
            "completed_updates": update,
            "train_loss": loss,
        }, trained

    def _receive_peer(self, vector, header):
        self._require_vector(vector)
        sender = int(header["sender_id"])
        with self.lock:
            self.peer_cache[sender] = vector.astype(np.float32, copy=True)
            self.peer_counters[sender] = int(header.get("sender_counter", 0))
        return {"ok": True, "client_id": self.client_id}

    def _push_current(self, header):
        with self.lock:
            vector = self.model.copy()
            counter = self.completed_updates
        peers = header.get("peers", [])
        message_type = header.get("message_type", "model_exchange")
        for peer in peers:
            self._peer_rpc(
                peer,
                {
                    "op": "receive_peer",
                    "sender_id": self.client_id,
                    "sender_counter": counter,
                    "message_type": message_type,
                },
                vector,
                message_type,
            )
        return {"ok": True, "client_id": self.client_id, "pushed_to": len(peers)}

    def _mix_train(self, header):
        weights = {int(key): float(value) for key, value in header["weights"].items()}
        with self.lock:
            vectors = []
            values = []
            for client_id, weight in weights.items():
                if client_id == self.client_id:
                    vectors.append(self.model.copy())
                elif client_id in self.peer_cache:
                    vectors.append(self.peer_cache[client_id].copy())
                else:
                    raise RuntimeError("missing model from peer %d" % client_id)
                values.append(weight)
        mixed = weighted_average(vectors, values)
        event = int(header.get("event", 0))
        started = time.time()
        trained, loss, batches = self._local_train(mixed, self._update_seed(event), event)
        ended = time.time()
        with self.lock:
            self.model = trained
            self.model_version += 1
            self.completed_updates += 1
            update = self.completed_updates
        self._log_update(started, ended, loss, batches, event, None, update)
        return {"ok": True, "client_id": self.client_id, "completed_updates": update, "train_loss": loss}

    def _start_worker(self, target, header):
        if self.worker is not None and self.worker.is_alive():
            raise RuntimeError("a background worker is already running")
        self.stop_event.clear()
        self.worker = threading.Thread(target=target, args=(dict(header),), daemon=True)
        self.worker.start()
        return {"ok": True, "client_id": self.client_id, "worker_started": True}

    def _adpsgd_loop(self, header):
        peers = list(header["peers"])
        weights = {int(key): float(value) for key, value in header["weights"].items()}
        max_skew = int(header.get("max_update_skew", -1))
        max_local_updates = int(header["max_local_updates"])
        self._push_current({"peers": peers, "message_type": "adpsgd_initial_model"})
        while not self.stop_event.is_set() and self.completed_updates < max_local_updates:
            if max_skew >= 0:
                with self.lock:
                    peer_values = list(self.peer_counters.values())
                    local_count = self.completed_updates
                if peer_values and local_count >= min(peer_values) + max_skew + 1:
                    time.sleep(0.01)
                    continue
            try:
                self._mix_train({"weights": weights, "event": self.completed_updates + 1})
                self._push_current({"peers": peers, "message_type": "adpsgd_model"})
            except Exception as exc:
                self._log({"kind": "error", "timestamp": time.time(), "where": "adpsgd_loop", "error": repr(exc)})
                time.sleep(0.05)

    def _cluster_loop(self, header):
        members = list(header["members"])
        leader_peers = list(header.get("leader_peers", []))
        cluster_id = int(header["cluster_id"])
        mixing_interval = int(header.get("mixing_interval", 1))
        max_cluster_events = int(header["max_cluster_events"])
        while not self.stop_event.is_set() and self.cluster_events < max_cluster_events:
            with self.lock:
                start_model = self.model.copy()
            event = self.cluster_events + 1
            vectors, weights, losses = [], [], []
            with ThreadPoolExecutor(max_workers=len(members)) as executor:
                futures = []
                for member in members:
                    if int(member["client_id"]) == self.client_id:
                        futures.append((member, executor.submit(self._train_self_for_cluster, start_model, event, cluster_id)))
                    else:
                        futures.append((member, executor.submit(self._train_peer_for_cluster, member, start_model, event, cluster_id)))
                for member, future in futures:
                    vector, loss, sample_count = future.result()
                    vectors.append(vector)
                    weights.append(sample_count)
                    if loss is not None:
                        losses.append(loss)
            aggregation_started = time.time()
            aggregate = weighted_average(vectors, weights)
            aggregation_ended = time.time()
            self._log_interval("AGGREGATION", aggregation_started, aggregation_ended, {"cluster_id": cluster_id, "event": event})
            with self.lock:
                self.model = aggregate
                self.model_version += 1
                self.cluster_events += 1
            if mixing_interval > 0 and event % mixing_interval == 0 and leader_peers:
                with self.lock:
                    available = [self.model.copy()]
                    mixing_weights = [1.0]
                    for peer in leader_peers:
                        peer_id = int(peer["client_id"])
                        if peer_id in self.peer_cache:
                            available.append(self.peer_cache[peer_id].copy())
                            mixing_weights.append(1.0)
                    self.model = weighted_average(available, mixing_weights)
                self._push_current({"peers": leader_peers, "message_type": "leader_mixing"})
            self._log(
                {
                    "kind": "cluster_event",
                    "timestamp": time.time(),
                    "cluster_id": cluster_id,
                    "event": event,
                    "members": len(members),
                    "mean_train_loss": float(np.mean(losses)) if losses else None,
                }
            )

    def _train_self_for_cluster(self, start_model, event, cluster_id):
        started = time.time()
        vector, loss, batches = self._local_train(start_model, self._update_seed(event), event)
        ended = time.time()
        with self.lock:
            self.completed_updates += 1
            update = self.completed_updates
        self._log_update(started, ended, loss, batches, event, cluster_id, update)
        return vector, loss, self.num_samples

    def _train_peer_for_cluster(self, member, start_model, event, cluster_id):
        response, vector, _ = self._peer_rpc(
            member,
            {"op": "train_from", "event": event, "cluster_id": cluster_id},
            start_model,
            "cluster_train",
        )
        return vector, response.get("train_loss"), int(response["num_samples"])

    def _peer_rpc(self, peer, header, vector, message_type):
        started = time.time()
        response, response_vector, stats = rpc(
            peer["hostname"],
            peer["port"],
            header,
            vector,
            timeout=float(self.run.get("socket_timeout_sec", 120.0)),
        )
        ended = time.time()
        self._log_interval("COMMUNICATION", started, ended, {"peer_id": int(peer["client_id"]), "message_type": message_type})
        self._log(
            {
                "kind": "message",
                "timestamp": ended,
                "sender_id": self.client_id,
                "peer_id": int(peer["client_id"]),
                "message_type": message_type,
                "category": "training",
                "request_bytes": stats["request_bytes"],
                "response_bytes": stats["response_bytes"],
                "transmitted_bytes": stats["total_bytes"],
                "elapsed_sec": stats["elapsed_sec"],
            }
        )
        with self.lock:
            self.training_bytes += int(stats["total_bytes"])
        return response, response_vector, stats

    def _local_train(self, start, seed, event):
        hidden_dims = self.manifest.get("hidden_dims", [int(self.manifest["hidden_dim"])])
        learning_rate = self._learning_rate(event)
        with self.training_lock:
            return train_local(
                start,
                self.x,
                self.y,
                int(self.manifest["input_dim"]),
                hidden_dims,
                int(self.manifest["num_classes"]),
                int(self.run["batch_size"]),
                int(self.run["local_epochs"]),
                learning_rate,
                float(self.run["momentum"]),
                float(self.run["weight_decay"]),
                int(seed),
            )

    def _learning_rate(self, event):
        initial = float(self.run["learning_rate"])
        if str(self.run.get("lr_schedule", "constant")).lower() != "cosine":
            return initial
        return cosine_learning_rate(
            initial,
            float(self.run.get("min_learning_rate", initial)),
            max(int(event), 1),
            int(self.run.get("rounds", 1)),
        )

    def _update_seed(self, event):
        return int(self.run.get("seed", 0)) * 1000003 + self.client_id * 1009 + int(event)

    def _status(self):
        with self.lock:
            return {
                "ok": True,
                "hostname": self.hostname,
                "client_id": self.client_id,
                "completed_updates": self.completed_updates,
                "cluster_events": self.cluster_events,
                "training_bytes": self.training_bytes,
                "model_version": self.model_version,
                "worker_alive": bool(self.worker is not None and self.worker.is_alive()),
                "stopped": self.stop_event.is_set(),
                "num_samples": self.num_samples,
            }

    def _flush_logs(self, output_dir):
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / ("%s_client%02d.jsonl" % (self.hostname, self.client_id))
        if self.local_log is not None and self.local_log.exists():
            target = destination / self.local_log.name
            shutil.copy2(str(self.local_log), str(target))
        return {"ok": True, "client_id": self.client_id, "log": str(target)}

    def _log_update(self, started, ended, loss, batches, event, cluster_id, update):
        self._log_interval("LOCAL_TRAIN", started, ended, {"event": event, "cluster_id": cluster_id, "update": update})
        self._log(
            {
                "kind": "update",
                "timestamp": ended,
                "client_id": self.client_id,
                "event": event,
                "cluster_id": cluster_id,
                "completed_updates": update,
                "local_train_sec": ended - started,
                "num_samples": self.num_samples,
                "num_batches": batches,
                "learning_rate": self._learning_rate(event),
                "train_loss": loss,
            }
        )

    def _log_interval(self, state, started, ended, extra=None):
        row = {"kind": "interval", "state": state, "start": started, "end": ended, "duration_sec": max(0.0, ended - started), "client_id": self.client_id}
        if extra:
            row.update(extra)
        self._log(row)

    def _log(self, row):
        if self.local_log is not None:
            row = dict(row)
            row.setdefault("run_id", self.run_id)
            row.setdefault("algorithm", self.run.get("algorithm"))
            append_jsonl(self.local_log, row)

    @staticmethod
    def _require_vector(vector):
        if vector is None:
            raise ValueError("operation requires a model payload")


class ThreadedServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        try:
            header, vector, _ = receive_packet(self.request)
            response, response_vector = self.server.agent.handle(header, vector)
            response = dict(response)
            response["ok"] = True
            send_packet(self.request, response, response_vector)
        except Exception as exc:
            try:
                send_packet(self.request, {"ok": False, "error": repr(exc), "traceback": traceback.format_exc()}, None)
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="Run one FIT IoT-LAB decentralized FL node agent.")
    parser.add_argument("--bundle-dir", default="/home/root/shared/iotlab_dfl/bundle")
    parser.add_argument("--config", default="/home/root/shared/iotlab_dfl/bundle/iotlab_config.json")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=29600)
    parser.add_argument("--client-id", type=int, default=None, help="Local smoke-test override.")
    args = parser.parse_args()
    agent = AgentState(args.bundle_dir, args.config, args.client_id)
    server = ThreadedServer((args.host, args.port), Handler)
    server.agent = agent
    print("[agent] host=%s client=%d listen=%s:%d" % (agent.hostname, agent.client_id, args.host, args.port), flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
