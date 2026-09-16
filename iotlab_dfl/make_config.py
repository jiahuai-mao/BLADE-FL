from __future__ import annotations

import argparse
import re
from pathlib import Path

from iotlab_dfl.storage import read_json, write_json


DEFAULT_NODE_IDS = "47-55+57-71+73-76"


def main():
    parser = argparse.ArgumentParser(description="Create FIT IoT-LAB node and training configuration.")
    parser.add_argument("--bundle-dir", default="iotlab_dfl/bundle")
    parser.add_argument("--node-ids", default=DEFAULT_NODE_IDS)
    parser.add_argument("--site", default="grenoble")
    parser.add_argument("--address-mode", choices=["hostname", "grenoble-ip"], default="hostname")
    parser.add_argument("--port", type=int, default=29600)
    parser.add_argument("--output", default="iotlab_dfl/bundle/iotlab_config.json")
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=10, help="Equivalent rounds between evaluations.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--min-lr", type=float, default=0.002)
    parser.add_argument("--lr-schedule", choices=["constant", "cosine"], default="cosine")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--mixing-interval", type=int, default=1)
    parser.add_argument("--random-edge-prob", type=float, default=0.45)
    parser.add_argument("--sparse-degree", type=int, default=6)
    args = parser.parse_args()
    manifest = read_json(Path(args.bundle_dir) / "manifest.json")
    ids = parse_ids(args.node_ids)
    if len(ids) != int(manifest["num_clients"]):
        raise ValueError("node list has %d entries; bundle has %d clients" % (len(ids), manifest["num_clients"]))
    nodes = []
    for client_id, node_id in enumerate(ids):
        nodes.append(
            {
                "client_id": client_id,
                "node_id": node_id,
                "hostname": (
                    "10.0.12.%d" % node_id
                    if args.address_mode == "grenoble-ip"
                    else "node-a8-%d.%s.iot-lab.info" % (node_id, args.site)
                ),
                "port": args.port,
            }
        )
    config = {
        "site": args.site,
        "nodes": nodes,
        "rounds": args.rounds,
        "target_client_updates": args.rounds * len(nodes),
        "eval_every_equivalent_rounds": args.eval_every,
        "batch_size": args.batch_size,
        "local_epochs": args.local_epochs,
        "learning_rate": args.lr,
        "min_learning_rate": args.min_lr,
        "lr_schedule": args.lr_schedule,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "mixing_interval": args.mixing_interval,
        "random_edge_probability": args.random_edge_prob,
        "sparse_degree": args.sparse_degree,
        "k_min": 2,
        "k_max": "sqrt",
        "min_clients_per_cluster": 2,
        "max_swaps": 10,
        "socket_timeout_sec": 120.0,
        "poll_interval_sec": 1.0,
    }
    write_json(args.output, config)
    print("[config] wrote %s with %d nodes" % (args.output, len(nodes)))


def parse_ids(value):
    result = []
    for part in re.split(r"[+,]", value):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = [int(item) for item in part.split("-", 1)]
            result.extend(range(left, right + 1))
        else:
            result.append(int(part))
    if len(result) != len(set(result)):
        raise ValueError("duplicate node ids")
    return result


if __name__ == "__main__":
    main()
