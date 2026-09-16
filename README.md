## Documents

- [Clustering algorithm requirements](docs/clustering_algorithm_requirements.md)
- [Training simulation requirements](docs/training_simulation_requirements.md)



## Basic Docker Commands (PowerShell)

```powershell
# 1) Build image
docker build --pull=false -t nc:dev -f ./Dockerfile .

# 2) Run container (first time)
# --shm-size prevents PyTorch DataLoader worker bus errors when num_workers > 0.
docker run -d --name nc-dev --runtime=nvidia --shm-size=16g -v "${PWD}:/workspace" -w /workspace nc:dev tail -f /dev/null

# 3) Start existing container
docker start nc-dev

# 4) Enter container
docker exec -it nc-dev bash

# 5) Cluster-only smoke test
docker exec adfed-dev python scripts/run_cluster_only.py --dataset synthetic --num-clients 20 --split dirichlet --k 2

