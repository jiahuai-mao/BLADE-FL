from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from iotlab_dfl.model import initialize_model, parameter_count
from iotlab_dfl.protocol import rpc
from iotlab_dfl.storage import write_json


def main():
    parser = argparse.ArgumentParser(description="Run a local multi-process smoke test of all IoT-LAB algorithms.")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()
    temporary = Path(tempfile.mkdtemp(prefix="iotlab-dfl-smoke-"))
    bundle = temporary / "bundle"
    results = temporary / "results"
    create_bundle(bundle)
    ports = reserve_ports(4)
    nodes = [{"client_id": idx, "node_id": idx, "hostname": "127.0.0.1", "port": ports[idx]} for idx in range(4)]
    config = {
        "site": "local",
        "nodes": nodes,
        "rounds": args.rounds,
        "target_client_updates": args.rounds * 4,
        "eval_every_equivalent_rounds": 1,
        "batch_size": 8,
        "local_epochs": 1,
        "learning_rate": 0.03,
        "momentum": 0.0,
        "weight_decay": 0.0,
        "mixing_interval": 1,
        "random_edge_probability": 0.6,
        "sparse_degree": 2,
        "k_min": 2,
        "k_max": "sqrt",
        "min_clients_per_cluster": 2,
        "max_swaps": 2,
        "socket_timeout_sec": 30.0,
        "poll_interval_sec": 0.05,
    }
    config_path = bundle / "iotlab_config.json"
    write_json(config_path, config)
    processes = []
    try:
        for client_id, port in enumerate(ports):
            processes.append(
                subprocess.Popen(
                    [sys.executable, "-m", "iotlab_dfl.agent", "--bundle-dir", str(bundle), "--config", str(config_path), "--host", "127.0.0.1", "--port", str(port), "--client-id", str(client_id)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            )
        wait_ready(nodes)
        for algorithm in ["sparse_dpsgd", "adpsgd", "bladefl", "mdfeel"]:
            command = [
                sys.executable,
                "-m",
                "iotlab_dfl.controller",
                "--algorithm",
                algorithm,
                "--seed",
                "0",
                "--rounds",
                str(args.rounds),
                "--bundle-dir",
                str(bundle),
                "--config",
                str(config_path),
                "--output-dir",
                str(results),
                "--node-output-dir",
                str(results),
                "--run-id",
                algorithm + "_seed0_smoke",
            ]
            print("[smoke] running %s" % algorithm, flush=True)
            subprocess.run(command, check=True)
        subprocess.run([sys.executable, "-m", "iotlab_dfl.collect_results", "--results-dir", str(results)], check=True)
        print("[smoke] passed; results=%s" % results)
    finally:
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        if args.keep:
            print("[smoke] retained %s" % temporary)


def create_bundle(bundle):
    data = bundle / "data"
    data.mkdir(parents=True)
    rng = np.random.RandomState(4)
    clients = []
    test_x, test_y = [], []
    for client_id in range(4):
        x = rng.randn(32, 12).astype(np.float32)
        y = ((x[:, 0] + 0.4 * x[:, 1] + client_id * 0.1) > 0).astype(np.int64)
        np.savez(data / ("client_%02d.npz" % client_id), x=x, y=y)
        clients.append({"client_id": client_id, "file": "data/client_%02d.npz" % client_id, "num_samples": len(y), "label_counts": np.bincount(y, minlength=2).tolist()})
        test_x.append(x[:8])
        test_y.append(y[:8])
    np.savez(data / "test.npz", x=np.concatenate(test_x), y=np.concatenate(test_y))
    hidden_dims = [8, 4]
    model = initialize_model(12, hidden_dims, 2, 0)
    np.save(bundle / "initial_model.npy", model)
    write_json(
        bundle / "manifest.json",
        {"dataset": "synthetic_smoke", "num_clients": 4, "num_classes": 2, "input_dim": 12, "hidden_dim": 8, "hidden_dims": hidden_dims, "parameter_count": parameter_count(12, hidden_dims, 2), "model_bytes": int(model.nbytes), "clients": clients},
    )


def reserve_ports(count):
    sockets, ports = [], []
    try:
        for _ in range(count):
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            sockets.append(sock)
            ports.append(sock.getsockname()[1])
    finally:
        for sock in sockets:
            sock.close()
    return ports


def wait_ready(nodes):
    deadline = time.time() + 20.0
    pending = list(nodes)
    while pending and time.time() < deadline:
        remaining = []
        for node in pending:
            try:
                rpc(node["hostname"], node["port"], {"op": "health"}, timeout=1.0)
            except Exception:
                remaining.append(node)
        pending = remaining
        time.sleep(0.05)
    if pending:
        raise RuntimeError("agents failed to start: %s" % pending)


if __name__ == "__main__":
    main()
