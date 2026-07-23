# conda run -n base python scripts/run_cluster_only.py --dataset synthetic --num-clients 20 --split dirichlet --k 4 --output-dir results/clustering
# conda run -n base python scripts/run_cluster_only.py --dataset synthetic --num-clients 20 --split dirichlet --k auto --output-dir results/clustering


# python scripts/run_training_simulation.py --algorithm clustered --dataset synthetic --num-clients 8 --split dirichlet --k auto --rounds 2 --eval-interval 1 --seed 0
# python scripts/run_training_simulation.py --algorithm dpsgd --dataset synthetic --num-clients 8 --split iid --rounds 2 --eval-interval 1 --seed 0
# python scripts/run_training_simulation.py --algorithm adpsgd --dataset synthetic --num-clients 8 --split iid --baseline-graph ring --events 8 --eval-interval 2 --seed 0
# python scripts/run_training_simulation.py --algorithm all --dataset synthetic --num-clients 8 --split dirichlet --k auto --rounds 2 --events 8 --eval-interval 1 --seed 0 


python scripts/run_training_simulation.py --algorithm all --dataset cifar10 --num-clients 50 --split dirichlet --dirichlet-alpha 0.1 --k auto --rounds 100 --events 500 --mixing-interval 5 --eval-interval 10 --seed 0
python scripts/run_training_simulation.py --algorithm all --dataset hhar --num-clients 50 --split dirichlet --dirichlet-alpha 0.1 --k auto --rounds 100 --events 500 --mixing-interval 5 --eval-interval 10 --seed 0
python scripts/summarize_main_performance.py --results-dir results/training --target-accuracy 0.70 --output-dir results/summary_main_performance