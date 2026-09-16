# FIT IoT-LAB physical HHAR validation

This package runs one logical client per physical A8 node using Python 3 and NumPy only. It is independent of the PyTorch simulator. Model payloads are exchanged directly between A8 nodes over TCP.

## Fixed experiment contract

- Physical scales: 28 and 56 clients.
- Methods: BLADE-FL, Sparse D-PSGD (`d=6`), AD-PSGD and adapted MD-FEEL.
- Training seeds: 0, 3 and 7.
- Budget: 300 equivalent rounds, or `300 * N` completed local updates.
- Model: 48-64-32-6 MLP with 5,414 parameters.
- Input: 48 accelerometer/gyroscope time-domain features from six-channel 128-sample HHAR windows.
- Optimizer: batch size 64, one local epoch, momentum 0.9 and weight decay 0.0001.
- Learning rate: cosine decay from 0.02 to 0.002.
- Evaluation: every 10 equivalent rounds on the same group-disjoint test set.

The train/test split is fixed with seed 0. Training groups are divided by activity and temporal continuity into blocks of at most 256 windows. Blocks are assigned using label-wise Dirichlet-guided preferences with `alpha=0.3` and partition seed 2026. Every client has at least two labels represented by at least 64 windows. The 28-client minimum size is 1,500 windows; the 56-client minimum is 750 windows. All methods share the frozen bundle, initial model and physical-node mapping within a scale.

## Prepare bundles locally

Run from the repository root:

```bash
sh iotlab_dfl/prepare_profiles.sh
```

This writes `bundle_28` and `bundle_56`. Their manifests include every temporal block assignment, source group, source index range, label histogram, entropy, JS divergence and assignment SHA-256.

Create a physical-node configuration only after IoT-LAB has allocated the nodes:

```bash
python -m iotlab_dfl.make_config \
  --bundle-dir iotlab_dfl/bundle_28 \
  --output iotlab_dfl/bundle_28/iotlab_config.json \
  --node-ids '47-55+57-71+73-76' \
  --address-mode grenoble-ip
```

For 56 clients, replace the bundle, output and node list with the 56 IDs returned by that reservation. `make_config` rejects a node count that differs from the bundle size.

## Local verification

```bash
python -m unittest discover -s iotlab_dfl/tests -v
python -m iotlab_dfl.smoke_local --rounds 2
```

## Clean upload

From the repository directory:

```bash
scp -r iotlab_dfl jmao@grenoble.iot-lab.info:shared/
```

The frontend and A8 nodes share this directory. Do not copy the package separately to individual nodes.

## Start one scale

On the Grenoble frontend:

```bash
cd ~/shared/iotlab_dfl
iotlab-ssh wait-for-boot --max-wait 600
sh deploy/check_environment.sh
sh deploy/start_agents.sh 28
sh deploy/start_grid.sh 28
sh deploy/status.sh 28
```

Use `56` for all four commands in a 56-node reservation. Only one profile may use a physical node set at a time.

The grid executes all 12 runs in the order seeds `0,3,7` and methods `bladefl,sparse_dpsgd,adpsgd,mdfeel`. Results are written to `results_28_r300` or `results_56_r300`. Completed runs are skipped on restart; an incomplete deterministic run directory stops the script for diagnosis.

## Retrieve and plot

```bash
python -m iotlab_dfl.collect_results --results-dir iotlab_dfl/results_28_r300
python -m iotlab_dfl.plot_physical_validation \
  --source-dir iotlab_dfl/results_28_r300/source_data \
  --output iotlab_dfl/results_28_r300/figure_physical_28
```

Repeat with `56`. Communication is the sum of application-layer request and response bytes for peer training RPCs plus algorithm-specific setup. Evaluation, monitoring and generic orchestration are logged separately and excluded. Wall-clock time includes topology profiling/construction and excludes synchronous evaluation pauses.

## Operational notes

- Agents listen on TCP port 29600 and log to `/tmp/iotlab_dfl_agent.log`.
- Per-run node logs are copied to the shared result directory at completion.
- Use `sh deploy/stop_agents.sh 28` or `56` before ending a reservation.
- Do not start two controllers against the same agents.
