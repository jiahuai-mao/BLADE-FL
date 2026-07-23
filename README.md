## Documents

- [Clustering algorithm requirements](docs/clustering_algorithm_requirements.md)
- [Training simulation requirements](docs/training_simulation_requirements.md)

## Cluster-Only Smoke Command

```powershell
python scripts/run_cluster_only.py --dataset synthetic --num-clients 20 --split dirichlet --k 2
python scripts/run_cluster_only.py --dataset synthetic --num-clients 20 --split dirichlet --k auto --k-min 2 --min-clients-per-cluster 5
```

For local datasets:

```powershell
python scripts/run_cluster_only.py --dataset mnist --data-dir data --num-clients 20 --split dirichlet --k 2
python scripts/run_cluster_only.py --dataset cifar10 --data-dir data --num-clients 20 --split dirichlet --k 2
python scripts/run_cluster_only.py --dataset cifar100 --data-dir data --num-clients 20 --split dirichlet --k 2
```

## Training Simulation Smoke Commands

The training simulation requires PyTorch. Use the Docker/conda environment defined by `environment.docker.yml`.

```powershell
python scripts/run_training_simulation.py --algorithm clustered --dataset synthetic --num-clients 8 --split dirichlet --k auto --rounds 2 --eval-interval 1 --seed 0
python scripts/run_training_simulation.py --algorithm fedavg --dataset synthetic --num-clients 8 --split iid --rounds 2 --eval-interval 1 --seed 0
python scripts/run_training_simulation.py --algorithm dpsgd --dataset synthetic --num-clients 8 --split iid --rounds 2 --eval-interval 1 --seed 0
python scripts/run_training_simulation.py --algorithm adpsgd --dataset synthetic --num-clients 8 --split iid --events 8 --eval-interval 2 --seed 0
python scripts/run_training_simulation.py --algorithm clustered_async_no_mixing --dataset synthetic --num-clients 8 --split dirichlet --k auto --events 8 --eval-interval 2 --seed 0
python scripts/run_training_simulation.py --algorithm random_clustered_async --dataset synthetic --num-clients 8 --split dirichlet --k auto --events 8 --eval-interval 2 --seed 0
python scripts/run_training_simulation.py --algorithm all --dataset synthetic --num-clients 8 --split dirichlet --k auto --rounds 2 --events 8 --eval-interval 1 --seed 0
```

For the main-performance experiment skeleton:

```powershell
python scripts/run_training_simulation.py --algorithm all --dataset cifar10 --num-clients 50 --split dirichlet --dirichlet-alpha 0.1 --k auto --rounds 100 --events 500 --mixing-interval 5 --eval-interval 10 --baseline-graph random --random-edge-prob 0.05 --seed 0
python scripts/run_training_simulation.py --algorithm all --dataset hhar --num-clients 50 --split dirichlet --dirichlet-alpha 0.1 --k auto --rounds 100 --events 500 --mixing-interval 5 --eval-interval 10 --baseline-graph random --random-edge-prob 0.05 --max-async-update-skew 1 --seed 0
python scripts/summarize_main_performance.py --results-dir results/training --target-accuracy 0.70 --output-dir results/summary_main_performance
```

## Basic Docker Commands (PowerShell)

```powershell
# 1) Build image
docker build --pull=false -t nc:dev -f .\Dockerfile .

# 2) Run container (first time)
# --shm-size prevents PyTorch DataLoader worker bus errors when num_workers > 0.
docker run -d --name nc-dev --gpus all --shm-size=8g -v "${PWD}:/workspace" -w /workspace nc:dev tail -f /dev/null

# 3) Start existing container
docker start adfed-dev

# 4) Enter container
docker exec -it nc-dev bash

# 5) Cluster-only smoke test
docker exec adfed-dev python scripts/run_cluster_only.py --dataset synthetic --num-clients 20 --split dirichlet --k 2


# 6) Rebuild (after Dockerfile/environment changes)
docker stop adfed-dev
docker rm adfed-dev
docker build --pull=false -t adfed:dev -f .\Dockerfile .    
docker run -d --name adfed-dev --gpus all --shm-size=8g -v "${PWD}:/workspace" -w /workspace adfed:dev tail -f /dev/null

# 7) Stop / remove
docker stop adfed-dev
docker rm adfed-dev
```
