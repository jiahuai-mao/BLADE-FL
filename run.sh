#!/usr/bin/env bash
set -euo pipefail

# python scripts/launch_main_performance.py \
#   --gpus 0,1,2 \
#   --jobs-per-gpu 3 \
#   --algorithm all \
#   --datasets cifar10 \
#   --alphas 0.1,0.3,0.5 \
#   --seeds 0,1,2 \
#   --k auto \
#   --k-max sqrt \
#   --rounds 500 \
#   --events 3500 \
#   --mixing-interval 1 \
#   --eval-interval 10 \
#   --baseline-graph random \
#   --random-edge-prob 0.45 \
#   -- \
#   --batch-size 512 \
#   --num-workers 2
#
# python scripts/launch_main_performance.py \
#   --gpus 0,1,2 \
#   --jobs-per-gpu 2 \
#   --algorithm clustered_async_clipprox \
#   --datasets cifar10 \
#   --alphas 0.1,0.3,0.5 \
#   --seeds 0,1 \
#   --k auto \
#   --k-max sqrt \
#   --rounds 500 \
#   --events 3500 \
#   --mixing-interval 1 \
#   --eval-interval 10 \
#   --baseline-graph random \
#   --random-edge-prob 0.45 \
#   --model resnet20 \
#   -- \
#   --num-workers 3 \
#   --clipprox-mu 0.001 \
#   --clipprox-clip-norm 5.0
#
# for algo in clustered_async clustered_async_clipprox dpsgd adpsgd; do
#   python scripts/launch_main_performance.py \
#     --gpus 0,1,2 \
#     --jobs-per-gpu 2 \
#     --algorithm "$algo" \
#     --datasets cifar10 \
#     --alphas 0.1,0.3,0.5 \
#     --seeds 0,3,7 \
#     --k auto \
#     --k-max sqrt \
#     --rounds 500 \
#     --events 3500 \
#     --mixing-interval 1 \
#     --eval-interval 50 \
#     --baseline-graph random \
#     --random-edge-prob 0.45 \
#     --model resnet20 \
#     -- \
#     --num-workers 3 \
#     --clipprox-mu 0.001 \
#     --clipprox-clip-norm 5.0
# done

# python scripts/launch_main_performance.py \
#   --gpus 0,1,2 \
#   --jobs-per-gpu 2 \
#   --algorithm dpsgd \
#   --datasets cifar10 \
#   --alphas 0.3,0.5 \
#   --seeds 0,3,7 \
#   --k auto \
#   --k-max sqrt \
#   --rounds 500 \
#   --events 3500 \
#   --mixing-interval 1 \
#   --eval-interval 50 \
#   --baseline-graph random \
#   --random-edge-prob 0.1224489796 \
#   --model resnet20 \
#   --output-dir results/training \
#   -- \
#   --num-workers 3 \
#   --clipprox-mu 0.001 \
#   --clipprox-clip-norm 5.0


# python scripts/launch_main_performance.py \
#   --gpus 0,1,2 \
#   --jobs-per-gpu 2 \
#   --algorithm mdfeel \
#   --datasets cifar10 \
#   --alphas 0.3,0.5 \
#   --seeds 0,3,7 \
#   --num-clients 50 \
#   --split dirichlet \
#   --k auto \
#   --k-max sqrt \
#   --rounds 500 \
#   --events 3500 \
#   --mixing-interval 1 \
#   --eval-interval 50 \
#   --baseline-graph random \
#   --random-edge-prob 0.45 \
#   --model resnet20 \
#   --batch-size 128 \
#   --local-epochs 1 \
#   --lr 0.05 \
#   --device cuda \
#   --output-dir results/training \
#   --log-dir results/logs/main_performance_mdfeel_cifar10 \
#   -- \
#   --num-workers 3






#
# ---------- HHAR Training ----------
#
# python scripts/launch_main_performance.py \
#   --gpus 0,1,2 \
#   --jobs-per-gpu 2 \
#   --algorithm core3 \
#   --datasets hhar \
#   --alphas 0.3 \
#   --seeds 0,3,7 \
#   --num-clients 101 \
#   --k auto \
#   --k-max sqrt \
#   --rounds 300 \
#   --events 2100 \
#   --mixing-interval 1 \
#   --eval-interval 50 \
#   --baseline-graph random \
#   --random-edge-prob 0.6 \
#   --model hhar1dcnn \
#   --hhar-feature-mode raw \
#   --hhar-window-size 128 \
#   --hhar-step-size 64 \
#   --hhar-sensor-mode acc_gyro \
#   --hhar-test-split window \
#   --hhar-client-partition group \
#   --batch-size 128 \
#   --local-epochs 1 \
#   --lr 0.01 \
#   --lr-scheduler cosine \
#   --momentum 0.9 \
#   --weight-decay 0.0005 \
#   -- \
#   --num-workers 0


# python scripts/launch_main_performance.py \
#   --gpus 0,1,2 \
#   --jobs-per-gpu 2 \
#   --algorithm dpsgd \
#   --datasets hhar \
#   --alphas 0.3 \
#   --seeds 0,3,7 \
#   --num-clients 101 \
#   --k auto \
#   --k-max sqrt \
#   --rounds 300 \
#   --events 2100 \
#   --mixing-interval 1 \
#   --eval-interval 50 \
#   --baseline-graph random \
#   --random-edge-prob 0.06 \
#   --model hhar1dcnn \
#   --hhar-feature-mode raw \
#   --hhar-window-size 128 \
#   --hhar-step-size 64 \
#   --hhar-sensor-mode acc_gyro \
#   --hhar-test-split window \
#   --hhar-client-partition group \
#   --batch-size 128 \
#   --local-epochs 1 \
#   --lr 0.01 \
#   --lr-scheduler cosine \
#   --momentum 0.9 \
#   --weight-decay 0.0005 \
#   --output-dir results/training \
#   -- \
#   --num-workers 0

# python scripts/launch_main_performance.py \
#   --gpus 0,1,2 \
#   --jobs-per-gpu 2 \
#   --algorithm mdfeel \
#   --datasets hhar \
#   --alphas 0.3 \
#   --seeds 0,3,7 \
#   --num-clients 101 \
#   --split dirichlet \
#   --k auto \
#   --k-max sqrt \
#   --rounds 300 \
#   --events 2100 \
#   --mixing-interval 1 \
#   --eval-interval 50 \
#   --baseline-graph random \
#   --random-edge-prob 0.6 \
#   --model hhar1dcnn \
#   --hhar-feature-mode raw \
#   --hhar-window-size 128 \
#   --hhar-step-size 64 \
#   --hhar-sensor-mode acc_gyro \
#   --hhar-test-split window \
#   --hhar-client-partition group \
#   --batch-size 128 \
#   --local-epochs 1 \
#   --lr 0.01 \
#   --lr-scheduler cosine \
#   --momentum 0.9 \
#   --weight-decay 0.0005 \
#   --device cuda \
#   --output-dir results/training \
#   --log-dir results/logs/main_performance_mdfeel_hhar \
#   -- \
#   --num-workers 0

#
# ---------- Figures ----------
#
# python scripts/plot_results1_all.py \
#   --results-dir results/training \
#   --output-dir results/figures/results1_nature_latest_p045_mix1 \
#   --dataset cifar10 \
#   --algorithm all \
#   --k-max sqrt \
#   --rounds 500 \
#   --events 3500 \
#   --random-edge-prob 0.45 \
#   --target-accuracy 0.30
#
# python scripts/plot_results1_core.py \
#   --results-dir results/training \
#   --output-dir results/figures/results1_nature_core_p045_mix1 \
#   --dataset cifar10 \
#   --algorithm all \
#   --k-max sqrt \
#   --rounds 500 \
#   --events 3500 \
#   --random-edge-prob 0.45 \
#   --target-accuracy 0.30
#
# python scripts/plot_nature_core3.py \
#   --results-dir results/training \
#   --output-dir results/figures/nature_core3_resnet20_eval50 \
#   --dataset cifar10 \
#   --model resnet20 \
#   --alphas 0.3,0.5 \
#   --seeds 0,3,7 \
#   --rounds 500 \
#   --events 3500 \
#   --eval-interval 50 \
#   --random-edge-prob 0.45


# ------------------------------------- Figure 4 --------------------------------------
# Figure 4 mixing ablations for CIFAR-10 alpha=0.3.
#
# This block writes to a dedicated Figure 4 directory so it does not overwrite
# the main results in results/training. Pass additional launcher arguments after
# run.sh, for example:
#
#   bash run.sh --dry-run
#   GPUS=0,1 JOBS_PER_GPU=1 bash run.sh
#
# Required runs for Fig. 4:
# - BLADE-FL / clustered_async with H = 1, 2, 5, 10, 20
# - No inter-cluster mixing / clustered_async_no_mixing
# - Synchronous leader mixing / clustered
# - D-PSGD as an optional decentralized reference
#
# Note: Fig. 4d requires per-evaluation cluster update counters. If that logging
# is not added before training, these runs still support Fig. 4a-c and 4e, but
# only final cluster counters will be available for Fig. 4d.
# ---------------------------------- run -----------------------------------------

# set -euo pipefail

# GPUS="${GPUS:-0,1,2}"
# JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
# DATA_DIR="${DATA_DIR:-data}"
# OUTPUT_DIR="${OUTPUT_DIR:-results/training_figure4_mixing_cifar10_a03}"
# LOG_DIR="${LOG_DIR:-results/logs_figure4_mixing_cifar10_a03}"
# RUN_SH_ARGS=("$@")

# FIG4_DATASET="cifar10"
# FIG4_ALPHAS="0.3"
# FIG4_SEEDS="0,3,7"
# FIG4_NUM_CLIENTS=50
# FIG4_ROUNDS=500
# FIG4_EVENTS=3500
# FIG4_EVAL_INTERVAL=50
# FIG4_RANDOM_EDGE_PROB=0.45
# FIG4_MODEL="resnet20"
# FIG4_BATCH_SIZE=128
# FIG4_LOCAL_EPOCHS=1
# FIG4_LR=0.05
# FIG4_LR_SCHEDULER="cosine"
# FIG4_NUM_WORKERS=3
# FIG4_SYNC_COMM_BANDWIDTH_PROXY="${FIG4_SYNC_COMM_BANDWIDTH_PROXY:-1000000}"
# FIG4_ALGORITHM_PLAN="${FIG4_ALGORITHM_PLAN:-clustered_async:1,clustered_async_no_mixing:1,clustered:1,dpsgd:1,clustered_async:2,clustered_async:5,clustered_async:10,clustered_async:20}"

# FIG4_COMMON_ARGS=(
#   --gpus "${GPUS}"
#   --jobs-per-gpu "${JOBS_PER_GPU}"
#   --datasets "${FIG4_DATASET}"
#   --alphas "${FIG4_ALPHAS}"
#   --seeds "${FIG4_SEEDS}"
#   --num-clients "${FIG4_NUM_CLIENTS}"
#   --split dirichlet
#   --k auto
#   --k-max sqrt
#   --rounds "${FIG4_ROUNDS}"
#   --events "${FIG4_EVENTS}"
#   --eval-interval "${FIG4_EVAL_INTERVAL}"
#   --baseline-graph random
#   --random-edge-prob "${FIG4_RANDOM_EDGE_PROB}"
#   --data-dir "${DATA_DIR}"
#   --output-dir "${OUTPUT_DIR}"
#   --log-dir "${LOG_DIR}"
#   --device cuda
#   --batch-size "${FIG4_BATCH_SIZE}"
#   --local-epochs "${FIG4_LOCAL_EPOCHS}"
#   --lr "${FIG4_LR}"
#   --lr-scheduler "${FIG4_LR_SCHEDULER}"
#   --sync-comm-bandwidth-proxy "${FIG4_SYNC_COMM_BANDWIDTH_PROXY}"
#   --momentum 0.9
#   --weight-decay 0.0005
#   --model "${FIG4_MODEL}"
# )

# FIG4_TRAIN_EXTRA_ARGS=(
#   --num-workers "${FIG4_NUM_WORKERS}"
# )

# echo "[fig4] Launching all Figure 4 jobs in one shared GPU queue"
# python scripts/launch_main_performance.py \
#   "${FIG4_COMMON_ARGS[@]}" \
#   "${RUN_SH_ARGS[@]}" \
#   --algorithm-plan "${FIG4_ALGORITHM_PLAN}" \
#   -- \
#   "${FIG4_TRAIN_EXTRA_ARGS[@]}"

# echo "[fig4] all requested Figure 4 runs are complete"


# GPUS=0,1,2 JOBS_PER_GPU=2 bash run.sh

# ------------------------------------- Figure 4 over --------------------------------------



# ---------------------------------------------------------------------------
# Figure plotting commands for the manuscript.
#
# These commands only regenerate paper figures from existing result files. They
# do not start training. They are kept commented so that running this run.sh
# does not accidentally regenerate figures after a training job. Uncomment the
# commands you need, or copy them to the terminal.
#
# Figure 2: Main performance and communication efficiency.
# Inputs:
#   - CIFAR-10 source data from results/figures/nature_core3_resnet20_eval50
#   - HHAR source data from results/figures/hhar_core3_nature
# Outputs:
#   - results/figures/subsection1/figure2_main_performance.{pdf,svg,png,tiff}
#   - results/figures/subsection1/figure2_source_*.csv
#
# MPLCONFIGDIR=/tmp/mplconfig FONTCONFIG_PATH=/tmp \
# python scripts/plot_subsection1.py \
#   --cifar-dir results/figures/nature_core3_resnet20_eval50 \
#   --hhar-dir results/figures/hhar_core3_nature \
#   --output-dir results/figures/subsection1
#
# Figure 3: Balanced topology construction.
# Inputs:
#   - training/shared topology artifacts under results/training
# Outputs:
#   - results/figures/subsection2/figure3_topology_balance.{pdf,svg,png,tiff}
#   - results/figures/subsection2/figure3_source_*.csv
#
# MPLCONFIGDIR=/tmp/mplconfig FONTCONFIG_PATH=/tmp \
# python scripts/plot_subsection2_topology.py \
#   --training-dir results/training \
#   --output-dir results/figures/subsection2 \
#   --random-baseline-repeats 20
#
# Figure 4: Leader-level mixing ablation.
# Inputs:
#   - Figure 4 mixing runs under results/training_figure4_mixing_cifar10_a03
#   - The current Figure 4 uses the original clustered synchronous baseline
#     for "Synchronous leader mixing"; clustered_barrier is not used here.
# Outputs:
#   - results/figures/subsection3/figure4_mixing_ablation.{pdf,svg,png,tiff}
#   - results/figures/subsection3/figure4_source_*.csv
#
# MPLCONFIGDIR=/tmp/mplconfig FONTCONFIG_PATH=/tmp \
# python scripts/plot_subsection3_mixing.py \
#   --training-dir results/training_figure4_mixing_cifar10_a03 \
#   --output-dir results/figures/subsection3 \
#   --representative-seed 0 \
#   --bytes-per-time-unit 1000000
#
# To regenerate all three figures in sequence, uncomment the three command
# blocks above in this section.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Figure 5: Dynamic participation with persistent client leave/join.
#
# Purpose:
#   - Fig. 5a: accuracy trajectory under client departure and join events.
#   - Fig. 5b: accuracy drop and recovery time after departure.
#
# Experimental design:
#   - CIFAR-10, Dirichlet alpha=0.3, ResNet20.
#   - Initial active clients: 50.
#   - Fixed full candidate-pool graph over 80 clients; leave/join changes the
#     active induced graph, not the underlying graph-generation process.
#   - Persistent departure ratios: 20%, 40%, 60%.
#   - Events: leave at 700, join at 1400, train until 2100.
#   - Policies: BLADE-FL refresh, stale topology, full reconstruction.
#   - Seeds: 0,3,7.
#   - Total jobs: 3 policies x 3 ratios x 3 seeds = 27.
#
# Outputs:
#   - results/training_figure5_dynamic_cifar10_a03/*/metrics.csv
#   - results/training_figure5_dynamic_cifar10_a03/*/event_summary.csv
#   - results/training_figure5_dynamic_cifar10_a03/*/topology_events.csv
#   - results/figures/subsection4/figure5_dynamic_participation.{pdf,svg,png,tiff}
#   - results/figures/subsection4/figure5_source_*.csv
#
# GPUS=0,1,2 JOBS_PER_GPU=2 bash run.sh is not used by this commented block.
# Copy the command below, or uncomment it and keep using bash rather than sh.
#
# python scripts/launch_dynamic_participation.py \
#   --gpus 0,1,2 \
#   --jobs-per-gpu 2 \
#   --dataset cifar10 \
#   --alphas 0.3 \
#   --seeds 0,3,7 \
#   --departure-ratios 0.2,0.4,0.6 \
#   --refresh-policies blade_refresh,stale_topology,full_reconstruction \
#   --num-clients 50 \
#   --pool-clients 80 \
#   --split dirichlet \
#   --k auto \
#   --k-max sqrt \
#   --events 2100 \
#   --leave-event 700 \
#   --join-event 1400 \
#   --mixing-interval 1 \
#   --eval-interval 50 \
#   --baseline-graph random \
#   --random-edge-prob 0.45 \
#   --data-dir data \
#   --output-dir results/training_figure5_dynamic_cifar10_a03 \
#   --log-dir results/logs_figure5_dynamic_cifar10_a03 \
#   --device cuda \
#   --model resnet20 \
#   --batch-size 128 \
#   --local-epochs 1 \
#   --lr 0.05 \
#   --lr-scheduler cosine \
#   --momentum 0.9 \
#   --weight-decay 0.0005 \
#   -- \
#   --num-workers 3
#
# After all Figure 5 training jobs complete, generate the figure:
#
# MPLCONFIGDIR=/tmp/mplconfig FONTCONFIG_PATH=/tmp \
# python scripts/plot_subsection4_dynamic.py \
#   --training-dir results/training_figure5_dynamic_cifar10_a03 \
#   --output-dir results/figures/subsection4 \
#   --representative-ratio 0.4 \
#   --x-axis event
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Figure 6: Scalability across client populations and topology overhead.
#
# Purpose:
#   - Fig. 6a: wall-clock proxy to target accuracy vs number of clients.
#   - Fig. 6b: communication to target accuracy vs number of clients.
#   - Fig. 6c: selected K and topology construction time vs number of clients.
#   - Fig. 6d: topology construction overhead relative to training wall time.
#
# Experimental design:
#   - CIFAR-10, Dirichlet alpha=0.3, ResNet20.
#   - Client populations: N = 50, 100, 200, 500.
#   - Algorithms: BLADE-FL, D-PSGD and AD-PSGD via --algorithm core3.
#   - Seeds: 0,3,7.
#   - Total jobs: 4 client scales x 3 algorithms x 3 seeds = 36.
#   - Results are saved to a new Figure 6 folder and do not overwrite previous
#     Figure 2/4/5 training outputs.
#
# Outputs:
#   - results/training_figure6_scalability_cifar10_a03/*/shared/cluster_metrics.json
#   - results/training_figure6_scalability_cifar10_a03/*/shared/k_search.json
#   - results/training_figure6_scalability_cifar10_a03/*/*/metrics.csv
#   - results/training_figure6_scalability_cifar10_a03/*/*/final_summary.json
#
# Copy and run these commands, or uncomment the blocks you need. Use bash,
# not sh, when launching multi-GPU jobs.
#
# python scripts/launch_main_performance.py \
#   --gpus 0,1,2 \
#   --jobs-per-gpu 2 \
#   --algorithm core3 \
#   --datasets cifar10 \
#   --alphas 0.3 \
#   --seeds 0,3,7 \
#   --num-clients 50 \
#   --split dirichlet \
#   --k auto \
#   --k-max sqrt \
#   --rounds 300 \
#   --events 2100 \
#   --mixing-interval 1 \
#   --eval-interval 50 \
#   --baseline-graph random \
#   --random-edge-prob 0.45 \
#   --model resnet20 \
#   --batch-size 128 \
#   --local-epochs 1 \
#   --lr 0.01 \
#   --lr-scheduler cosine \
#   --momentum 0.9 \
#   --weight-decay 0.0005 \
#   --output-dir results/training_figure6_scalability_cifar10_a03 \
#   --log-dir results/logs_figure6_scalability_cifar10_a03 \
#   -- \
#   --num-workers 0

# python scripts/launch_main_performance.py \
#   --gpus 0,1,2 \
#   --jobs-per-gpu 2 \
#   --algorithm core3 \
#   --datasets cifar10 \
#   --alphas 0.3 \
#   --seeds 0,3,7 \
#   --num-clients 100 \
#   --split dirichlet \
#   --k auto \
#   --k-max sqrt \
#   --rounds 300 \
#   --events 2100 \
#   --mixing-interval 1 \
#   --eval-interval 50 \
#   --baseline-graph random \
#   --random-edge-prob 0.45 \
#   --model resnet20 \
#   --batch-size 128 \
#   --local-epochs 1 \
#   --lr 0.01 \
#   --lr-scheduler cosine \
#   --momentum 0.9 \
#   --weight-decay 0.0005 \
#   --output-dir results/training_figure6_scalability_cifar10_a03 \
#   --log-dir results/logs_figure6_scalability_cifar10_a03 \
#   -- \
#   --num-workers 0

# python scripts/launch_main_performance.py \
#   --gpus 0,1,2 \
#   --jobs-per-gpu 2 \
#   --algorithm core3 \
#   --datasets cifar10 \
#   --alphas 0.3 \
#   --seeds 0,3,7 \
#   --num-clients 200 \
#   --split dirichlet \
#   --k auto \
#   --k-max sqrt \
#   --rounds 300 \
#   --events 2100 \
#   --mixing-interval 1 \
#   --eval-interval 50 \
#   --baseline-graph random \
#   --random-edge-prob 0.45 \
#   --model resnet20 \
#   --batch-size 128 \
#   --local-epochs 1 \
#   --lr 0.01 \
#   --lr-scheduler cosine \
#   --momentum 0.9 \
#   --weight-decay 0.0005 \
#   --output-dir results/training_figure6_scalability_cifar10_a03 \
#   --log-dir results/logs_figure6_scalability_cifar10_a03 \
#   -- \
#   --num-workers 0

# python scripts/launch_main_performance.py \
#   --gpus 0,1,2 \
#   --jobs-per-gpu 1 \
#   --algorithm adpsgd \
#   --datasets cifar10 \
#   --alphas 0.3 \
#   --seeds 0,3,7 \
#   --num-clients 500 \
#   --split dirichlet \
#   --k auto \
#   --k-max sqrt \
#   --rounds 300 \
#   --events 2100 \
#   --mixing-interval 1 \
#   --eval-interval 50 \
#   --baseline-graph random \
#   --random-edge-prob 0.05 \
#   --model resnet20 \
#   --batch-size 128 \
#   --local-epochs 1 \
#   --lr 0.01 \
#   --lr-scheduler cosine \
#   --momentum 0.9 \
#   --weight-decay 0.0005 \
#   --output-dir results/training_figure6_scalability_cifar10_a03 \
#   --log-dir results/logs_figure6_scalability_adpsgd_only_cifar10_a03 \
#   -- \
#   --num-workers 0

# python scripts/launch_main_performance.py \
#   --gpus 2 \
#   --jobs-per-gpu 1 \
#   --algorithm clustered_async \
#   --datasets cifar10 \
#   --alphas 0.3 \
#   --seeds 3 \
#   --num-clients 200 \
#   --split dirichlet \
#   --k auto \
#   --k-max sqrt \
#   --rounds 600 \
#   --events 4200 \
#   --mixing-interval 1 \
#   --eval-interval 100 \
#   --baseline-graph random \
#   --random-edge-prob 0.45 \
#   --model resnet20 \
#   --batch-size 128 \
#   --local-epochs 1 \
#   --lr 0.01 \
#   --lr-scheduler cosine \
#   --momentum 0.9 \
#   --weight-decay 0.0005 \
#   --output-dir results/training_figure6_scalability_cifar10_a03_long75_retry \
#   --log-dir results/logs_figure6_scalability_cifar10_a03_long75_retry \
#   -- \
#   --num-workers 0


# python scripts/plot_subsection5_scalability.py \
#   --training-dir results/training_figure6_scalability_cifar10_a03 \
#   --output-dir results/figures/subsection5 \
#   --target-accuracy 0.50 \
#   --bytes-per-time-unit 1000000
# ---------------------------------------------------------------------------


# python scripts/launch_main_performance.py \
#   --gpus 2 \
#   --jobs-per-gpu 1 \
#   --algorithm adpsgd \
#   --datasets cifar10 \
#   --alphas 0.3 \
#   --seeds 0,3,7 \
#   --num-clients 200 \
#   --split dirichlet \
#   --k auto \
#   --k-max sqrt \
#   --rounds 4800 \
#   --events 33600 \
#   --mixing-interval 1 \
#   --eval-interval 100 \
#   --baseline-graph random \
#   --random-edge-prob 0.45 \
#   --model resnet20 \
#   --batch-size 128 \
#   --local-epochs 1 \
#   --lr 0.01 \
#   --lr-scheduler cosine \
#   --momentum 0.9 \
#   --weight-decay 0.0005 \
#   --output-dir results/training_figure6_scalability_cifar10_a03_baselines75 \
#   --log-dir results/logs_figure6_scalability_cifar10_a03_baselines75 \
#   -- \
#   --num-workers 0


# ---------------------------------------------------------------------------
# Figure 2 CIFAR-10 cosine reruns.
#
# Purpose:
# - Regenerate the Figure 2 CIFAR-10 comparison with the same cosine learning-
#   rate schedule used by the Figure 4 mechanism experiment.
# - Cover alpha = 0.3 and 0.5, seeds 0,3,7, and all five Figure 2 methods.
# - Replace only the corresponding Figure 2 algorithm results in results/training.
#   Other datasets, ablations and algorithm variants remain untouched.
#
# Job count:
# - 24 jobs at p=0.45: BLADE-FL, D-PSGD, AD-PSGD and adapted MD-FEEL.
# - 6 sparse D-PSGD jobs at p=6/(50-1).
#
# WARNING: --force-rerun removes each targeted algorithm result directory before
# training it again. This is the only active experiment block in run.sh.
# ---------------------------------------------------------------------------

FIG2_COSINE_OUTPUT="${FIG2_COSINE_OUTPUT:-results/training}"
FIG2_COSINE_LOGS="${FIG2_COSINE_LOGS:-results/logs_figure2_cifar10_cosine}"
GPUS="${GPUS:-0,1,2}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"

# BLADE-FL, dense D-PSGD, AD-PSGD and adapted MD-FEEL.
python scripts/launch_main_performance.py \
  --gpus "${GPUS}" \
  --jobs-per-gpu "${JOBS_PER_GPU}" \
  --algorithm-plan clustered_async:1,dpsgd:1,adpsgd:1,mdfeel:1 \
  --datasets cifar10 \
  --alphas 0.3,0.5 \
  --seeds 0,3,7 \
  --num-clients 50 \
  --split dirichlet \
  --k auto \
  --k-max sqrt \
  --rounds 500 \
  --events 3500 \
  --mixing-interval 1 \
  --eval-interval 50 \
  --baseline-graph random \
  --random-edge-prob 0.45 \
  --model resnet20 \
  --batch-size 128 \
  --local-epochs 1 \
  --lr 0.05 \
  --lr-scheduler cosine \
  --cluster-delta-clip-norm 0.0 \
  --momentum 0.9 \
  --weight-decay 0.0005 \
  --device cuda \
  --data-dir data \
  --output-dir "${FIG2_COSINE_OUTPUT}" \
  --log-dir "${FIG2_COSINE_LOGS}/main" \
  -- \
  --force-rerun \
  --num-workers 3

# Sparse D-PSGD with expected graph degree d=6: p=6/(50-1).
python scripts/launch_main_performance.py \
  --gpus "${GPUS}" \
  --jobs-per-gpu "${JOBS_PER_GPU}" \
  --algorithm dpsgd \
  --datasets cifar10 \
  --alphas 0.3,0.5 \
  --seeds 0,3,7 \
  --num-clients 50 \
  --split dirichlet \
  --k auto \
  --k-max sqrt \
  --rounds 500 \
  --events 3500 \
  --mixing-interval 1 \
  --eval-interval 50 \
  --baseline-graph random \
  --random-edge-prob 0.12244897959183673 \
  --model resnet20 \
  --batch-size 128 \
  --local-epochs 1 \
  --lr 0.05 \
  --lr-scheduler cosine \
  --cluster-delta-clip-norm 0.0 \
  --momentum 0.9 \
  --weight-decay 0.0005 \
  --device cuda \
  --data-dir data \
  --output-dir "${FIG2_COSINE_OUTPUT}" \
  --log-dir "${FIG2_COSINE_LOGS}/sparse_dpsgd" \
  -- \
  --force-rerun \
  --num-workers 3


# ---------------------------------------------------------------------------
# BLADE-FLv1 reruns: same configs as BLADE-FL, 3 GPUs, 1 job per GPU.
#
# Notes:
# - BLADE-FLv1 uses algorithm name "clustered_asyncv1".
# - Outputs are saved beside the original BLADE-FL results, with run names like
#   clustered_asyncv1_..., so the existing clustered_async_* results are not
#   overwritten.
# - Figure 3 is topology-only and does not require retraining.
# - Figure 5 dynamic participation needs a separate dynamic-runner v1 hook if
#   you want to rerun that panel with BLADE-FLv1.
# ---------------------------------------------------------------------------

# # Figure 2: CIFAR-10 main-performance BLADE-FLv1
# python scripts/launch_main_performance.py \
#   --gpus 0,1,2 \
#   --jobs-per-gpu 1 \
#   --algorithm clustered_asyncv1 \
#   --datasets cifar10 \
#   --alphas 0.3\
#   --seeds 0,3,7 \
#   --num-clients 50 \
#   --split dirichlet \
#   --k auto \
#   --k-max sqrt \
#   --rounds 500 \
#   --events 3500 \
#   --mixing-interval 1 \
#   --eval-interval 50 \
#   --baseline-graph random \
#   --random-edge-prob 0.45 \
#   --model resnet20 \
#   --batch-size 128 \
#   --local-epochs 1 \
#   --lr 0.05 \
#   --lr-scheduler cosine \
#   --cluster-delta-clip-norm 2.0 \
#   --momentum 0.9 \
#   --weight-decay 0.0005 \
#   --device cuda \
#   --output-dir results/training \
#   --log-dir results/logs/main_performance_bladeflv1_cifar10 \
#   -- \
#   --num-workers 3

# Figure 2: HHAR main-performance BLADE-FLv1
# python scripts/launch_main_performance.py \
#   --gpus 0,1,2 \
#   --jobs-per-gpu 1 \
#   --algorithm clustered_asyncv1 \
#   --datasets hhar \
#   --alphas 0.3 \
#   --seeds 0,3,7 \
#   --num-clients 101 \
#   --split dirichlet \
#   --k auto \
#   --k-max sqrt \
#   --rounds 300 \
#   --events 2100 \
#   --mixing-interval 1 \
#   --eval-interval 50 \
#   --baseline-graph random \
#   --random-edge-prob 0.6 \
#   --model hhar1dcnn \
#   --hhar-feature-mode raw \
#   --hhar-window-size 128 \
#   --hhar-step-size 64 \
#   --hhar-sensor-mode acc_gyro \
#   --hhar-test-split window \
#   --hhar-client-partition group \
#   --batch-size 128 \
#   --local-epochs 1 \
#   --lr 0.01 \
#   --lr-scheduler cosine \
#   --cluster-delta-clip-norm 1.0 \
#   --momentum 0.9 \
#   --weight-decay 0.0005 \
#   --device cuda \
#   --output-dir results/training \
#   --log-dir results/logs/main_performance_bladeflv1_hhar \
#   -- \
#   --num-workers 0

# # Figure 4: mixing interval ablation BLADE-FLv1, H = 1,2,5,10,20
# python scripts/launch_main_performance.py \
#   --gpus 0,1,2 \
#   --jobs-per-gpu 1 \
#   --algorithm-plan clustered_asyncv1:1,clustered_asyncv1:2,clustered_asyncv1:5,clustered_asyncv1:10,clustered_asyncv1:20 \
#   --datasets cifar10 \
#   --alphas 0.3 \
#   --seeds 0,3,7 \
#   --num-clients 50 \
#   --split dirichlet \
#   --k auto \
#   --k-max sqrt \
#   --rounds 500 \
#   --events 3500 \
#   --mixing-interval 1 \
#   --eval-interval 50 \
#   --baseline-graph random \
#   --random-edge-prob 0.45 \
#   --model resnet20 \
#   --batch-size 128 \
#   --local-epochs 1 \
#   --lr 0.05 \
#   --lr-scheduler none \
#   --cluster-delta-clip-norm 1.0 \
#   --momentum 0.9 \
#   --weight-decay 0.0005 \
#   --device cuda \
#   --output-dir results/training_figure4_mixing_cifar10_a03 \
#   --log-dir results/logs_figure4_mixing_cifar10_a03_bladeflv1 \
#   -- \
#   --num-workers 3

# # Figure 6: scalability BLADE-FLv1, N=50, seeds 0/3/7
# python scripts/launch_main_performance.py \
#   --gpus 0,1,2 \
#   --jobs-per-gpu 1 \
#   --algorithm clustered_asyncv1 \
#   --datasets cifar10 \
#   --alphas 0.3 \
#   --seeds 0,3,7 \
#   --num-clients 50 \
#   --split dirichlet \
#   --k auto \
#   --k-max sqrt \
#   --rounds 600 \
#   --events 4200 \
#   --mixing-interval 1 \
#   --eval-interval 100 \
#   --baseline-graph random \
#   --random-edge-prob 0.45 \
#   --model resnet20 \
#   --batch-size 128 \
#   --local-epochs 1 \
#   --lr 0.01 \
#   --lr-scheduler cosine \
#   --cluster-delta-clip-norm 1.0 \
#   --momentum 0.9 \
#   --weight-decay 0.0005 \
#   --device cuda \
#   --output-dir results/training_figure6_scalability_cifar10_a03_long75 \
#   --log-dir results/logs_figure6_scalability_cifar10_a03_long75_bladeflv1_n50 \
#   -- \
#   --num-workers 0

# # Figure 6: scalability BLADE-FLv1, N=100/200, seeds 0/7
# python scripts/launch_main_performance.py \
#   --gpus 0,1,2 \
#   --jobs-per-gpu 1 \
#   --algorithm clustered_asyncv1 \
#   --datasets cifar10 \
#   --alphas 0.3 \
#   --seeds 0,7 \
#   --num-clients 100,200 \
#   --split dirichlet \
#   --k auto \
#   --k-max sqrt \
#   --rounds 600 \
#   --events 4200 \
#   --mixing-interval 1 \
#   --eval-interval 100 \
#   --baseline-graph random \
#   --random-edge-prob 0.45 \
#   --model resnet20 \
#   --batch-size 128 \
#   --local-epochs 1 \
#   --lr 0.01 \
#   --lr-scheduler cosine \
#   --cluster-delta-clip-norm 1.0 \
#   --momentum 0.9 \
#   --weight-decay 0.0005 \
#   --device cuda \
#   --output-dir results/training_figure6_scalability_cifar10_a03_long75 \
#   --log-dir results/logs_figure6_scalability_cifar10_a03_long75_bladeflv1_n100_n200_seed0_seed7 \
#   -- \
#   --num-workers 0

# # Figure 6: scalability BLADE-FLv1, N=100/200, seed 3 retry-long setting
# python scripts/launch_main_performance.py \
#   --gpus 0,1,2 \
#   --jobs-per-gpu 1 \
#   --algorithm clustered_asyncv1 \
#   --datasets cifar10 \
#   --alphas 0.3 \
#   --seeds 3 \
#   --num-clients 100,200 \
#   --split dirichlet \
#   --k auto \
#   --k-max sqrt \
#   --rounds 800 \
#   --events 5600 \
#   --mixing-interval 1 \
#   --eval-interval 100 \
#   --baseline-graph random \
#   --random-edge-prob 0.45 \
#   --model resnet20 \
#   --batch-size 128 \
#   --local-epochs 1 \
#   --lr 0.01 \
#   --lr-scheduler cosine \
#   --cluster-delta-clip-norm 1.0 \
#   --momentum 0.9 \
#   --weight-decay 0.0005 \
#   --device cuda \
#   --output-dir results/training_figure6_scalability_cifar10_a03_long75_retry \
#   --log-dir results/logs_figure6_scalability_cifar10_a03_long75_retry_bladeflv1 \
#   -- \
#   --num-workers 0
