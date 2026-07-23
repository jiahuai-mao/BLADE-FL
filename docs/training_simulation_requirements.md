# 训练模拟与聚合算法需求文档

## 1. 目标

本模块用于在单设备上模拟多个异构客户端的训练过程，并在同一实验框架下比较：

1. 分簇训练算法。
2. D-PSGD。
3. AD-PSGD。
4. 后续可扩展的其他 decentralized / clustered aggregation algorithms。

训练模拟必须复用分簇模块输出的 `Topology`：

```python
Topology(
    clusters: list[list[int]],
    leaders: list[int],
    leader_edges: list[tuple[int, int]],
    client_to_cluster: dict[int, int],
)
```

分簇算法只负责生成 topology；训练模块只消费 topology，不修改分簇目标函数。

## 2. 非目标

第一阶段不做以下内容：

1. 不模拟通信带宽或上传下载耗时。
2. 不使用真实 wall-clock 时间作为客户端训练时间。
3. 不做动态 join/leave。
4. 不做真实多进程或多设备并行训练。
5. 不引入未定义的额外分簇目标。

## 3. 客户端模拟

每个 client 由三部分组成：

```python
ClientState(
    client_id: int,
    train_loader: DataLoader,
    metadata: ClientMetadata,
    model_vector: Tensor,
    local_step: int,
)
```

其中 `metadata` 来自分簇阶段：

```python
ClientMetadata(
    client_id: int,
    num_samples: int,
    label_counts: list[int],
    train_time: float,
)
```

当前阶段的虚拟本地训练时间：

```text
T_i^loc = local_epochs * n_i
```

若 `local_epochs=1`，则：

```text
T_i^loc = n_i
```

该时间只用于模拟异步进度和 staleness，不包含通信时间。

## 4. 数据集与划分

训练模块需要支持：

1. `synthetic`
2. `mnist`
3. `fashionmnist`
4. `cifar10`
5. `cifar100`
6. `svhn`

数据划分方式：

```yaml
data:
  split: iid | dirichlet
  dirichlet_alpha: 0.5
  num_clients: 100
  samples_fraction: 1.0
```

所有随机操作必须由 `seed` 控制。

## 5. 统一训练接口

建议定义统一 runner 接口：

```python
class TrainingRunner:
    def run(self) -> RunArtifacts:
        ...
```

输出：

```python
RunArtifacts(
    metrics: list[dict],
    summary: dict,
)
```

每种算法实现一个 runner：

```text
ClusteredTrainingRunner
DPSGDRunner
ADPSGDRunner
```

## 6. 模型状态表示

所有算法内部统一使用 flattened model vector 表示模型状态：

```python
model_vector = parameters_to_vector(model.parameters())
```

需要提供工具函数：

```python
model_to_vector(model) -> Tensor
vector_to_model(vector, model) -> Module
average_vectors(vectors, weights) -> Tensor
```

这样 Clustered、D-PSGD、AD-PSGD 可以共享模型聚合逻辑。

## 7. 分簇训练算法

### 7.1 输入

```python
clients: list[ClientState]
topology: Topology
leader_edges: list[tuple[int, int]]
```

### 7.2 每个 cluster 的本地训练

对 cluster `V_k`：

1. 每个 client 从当前 cluster model 初始化。
2. 每个 client 本地训练 `local_epochs`。
3. 得到本地模型变化：

```text
Delta_i = m_i^local - m_k
```

4. leader 按样本数加权聚合：

```text
Delta_k =
sum_{i in V_k}
(n_i / n_k) Delta_i
```

5. 更新 cluster model：

```text
tilde_m_k = m_k + Delta_k
```

该过程对应簇内同步训练。簇内完成时间由虚拟时间决定：

```text
cluster_update_time(k) = max_{i in V_k} T_i^loc
```

### 7.3 簇间 mixing

每个 cluster 维护本地计数器：

```text
t_k
```

每完成 `H` 次 cluster-local update 后，触发一次 leader overlay mixing。

leader overlay 由 `topology.leader_edges` 给出。

默认 mixing matrix 使用 Metropolis 权重：

```text
w_kj = 1 / (1 + max(deg(k), deg(j)))   if (k,j) in leader_edges
w_kk = 1 - sum_j w_kj
w_kj = 0 otherwise
```

### 7.4 同步 clustered baseline

第一版可先实现同步 clustered training：

1. 所有 cluster 每一 global round 都完成一次本地更新。
2. 每个 global round 后执行一次或按 `H` 执行 mixing。
3. 不使用真实异步事件队列。

### 7.5 异步 clustered training

第二版实现事件驱动模拟。

为每个 cluster 维护：

```python
next_finish_time[k]
cluster_counter[k]
cached_neighbor_state[k][j]
cached_neighbor_counter[k][j]
```

事件循环：

1. 选择 `next_finish_time` 最小的 cluster `k`。
2. cluster `k` 执行一次本地更新。
3. `cluster_counter[k] += 1`。
4. 如果 `cluster_counter[k] % H == 0`，执行 mixing。
5. 将新 state 发送到邻居缓存。
6. 更新：

```text
next_finish_time[k] += cluster_update_time(k)
```

不使用通信时间；缓存更新可以在发送时立即可见，或后续加入固定/随机消息延迟。

## 8. D-PSGD

D-PSGD 不使用 cluster topology，而是把所有 client 作为 decentralized graph 节点。

### 8.1 输入 graph

```yaml
baseline:
  graph_type: full | ring | random
  random_edge_prob: 0.3
```

D-PSGD 和 AD-PSGD 必须在同一次运行中共享同一个 baseline graph。
默认 baseline graph 使用 `random`，并由 `seed` 和 `random_edge_prob`
确定；`full` 只用于额外 sanity check，不作为主实验默认设置。

### 8.2 每轮流程

每个 client `i`：

1. 从上一轮本地模型 `m_i^r` 开始。
2. 根据 graph 的 Metropolis matrix 同步 mixing：

```text
bar_m_i^r = sum_j w_ij m_j^r
```

3. 从 `bar_m_i^r` 初始化本地模型。
4. 本地训练 `local_epochs`。
5. 得到 `m_i^{r+1}`。

D-PSGD 是同步轮式模拟：

```text
所有 client 在同一 round 使用同一 reference state
```

虚拟 round time 可记录为：

```text
round_time = max_i T_i^loc
```

但该时间只用于 metrics，不改变同步算法更新顺序。

## 9. AD-PSGD

AD-PSGD 使用事件驱动异步模拟。

每个 client 维护：

```python
client_model[i]
client_counter[i]
next_finish_time[i]
neighbor_cache[i][j]
neighbor_counter_cache[i][j]
```

事件循环：

1. 选择 `next_finish_time` 最小的 client `i`。
2. client `i` 使用当前缓存中的邻居模型做 mixing：

```text
bar_m_i = w_ii m_i + sum_{j in N_i} w_ij cached_m_{j->i}
```

3. 从 `bar_m_i` 本地训练。
4. 更新 `client_model[i]`。
5. `client_counter[i] += 1`。
6. 将新模型写入邻居缓存。
7. 更新：

```text
next_finish_time[i] += T_i^loc
```

### 9.1 staleness

对 client `i` 使用邻居 `j` 缓存时：

```text
staleness_{i,j} = client_counter[j] - cached_counter[i][j]
```

记录：

```text
mean_staleness
max_staleness
```

第一阶段不加入通信延迟，因此 staleness 主要来自不同 client 虚拟训练速度不同。

## 10. 虚拟时间系统

训练模拟必须显式维护 virtual time。

同步算法：

```text
elapsed_virtual_time += max_i T_i^loc
```

分簇同步算法：

```text
elapsed_virtual_time += max_k cluster_update_time(k)
```

异步算法：

```text
elapsed_virtual_time = current_event_time
```

虚拟时间必须写入 metrics：

```python
{
    "round_or_event": int,
    "virtual_time": float,
    "algorithm": str,
}
```

## 11. 评估方式

需要支持统一评估：

1. global test accuracy。
2. global test loss。
3. best accuracy。
4. virtual time。
5. transmitted bytes proxy。
6. mean/max staleness。
7. model divergence。

### 11.1 global model 构造

Clustered training：

```text
global_model = sample-weighted average of cluster models
```

D-PSGD / AD-PSGD：

```text
global_model = sample-weighted average of client models
```

### 11.2 评估间隔

```yaml
training:
  eval_interval: 10
```

异步算法中可以按 event 数或 virtual time 间隔评估，第一版按 event 数评估。

## 12. 输出文件

建议目录：

```text
results/training/{run_name}/
  config.json
  topology.json                 # clustered algorithms only
  k_search.json                 # k=auto only
  metrics.csv
  final_summary.json
```

`metrics.csv` 字段：

```text
step
algorithm
K
virtual_time
train_loss
test_loss
test_accuracy
best_accuracy
mean_staleness
max_staleness
leader_edges
transmitted_bytes_proxy
```

## 13. 配置建议

```yaml
experiment:
  seed: 0
  output_dir: results/training

data:
  dataset: cifar10
  data_dir: data
  split: dirichlet
  dirichlet_alpha: 0.5
  num_clients: 100
  samples_fraction: 1.0

clustering:
  enabled: true
  k: auto
  k_min: 2
  min_clients_per_cluster: 5
  alpha_time: 0.5
  alpha_distribution: 0.5
  lambda_communication: 0.2
  mixing_interval: 5
  max_swaps: 10

training:
  algorithm: clustered | dpsgd | adpsgd
  rounds: 100
  events: 1000
  local_epochs: 1
  batch_size: 64
  lr: 0.01
  optimizer: sgd
  momentum: 0.0
  weight_decay: 0.0
  eval_interval: 10
  device: cuda

baseline_graph:
  graph_type: random
  random_edge_prob: 0.3

simulation:
  time_mode: local_epochs_num_samples
  use_virtual_time: true
  communication_delay: none
```

## 14. 可复现性

所有随机来源必须由 `seed` 控制：

1. Python `random`。
2. NumPy。
3. PyTorch CPU。
4. PyTorch CUDA。
5. DataLoader shuffle generator。
6. Dirichlet data partition。
7. random graph generation。

同一配置重复运行，以下结果应一致：

```text
client split
topology
leader_edges
initial model
metrics trajectory
```

## 15. 建议代码结构

```text
clustered_dfl/
  types.py
  data.py
  clustering.py
  topology.py
  models.py
  training/
    __init__.py
    vector_utils.py
    graph.py
    time.py
    metrics.py
    clustered.py
    dpsgd.py
    adpsgd.py
scripts/
  run_training_simulation.py
```

## 16. 第一阶段实现顺序

建议按以下顺序实现：

1. 训练数据加载和 client DataLoader 构建。
2. 简单模型，例如 MNIST CNN / CIFAR CNN。
3. vector 工具函数。
4. D-PSGD 同步 runner。
5. 分簇同步 runner。
6. AD-PSGD 事件驱动 runner。
7. clustered asynchronous runner。
8. 统一 metrics 和 plotting。

## 17. 验收标准

### 17.1 D-PSGD

1. random graph 下所有 client 可同步 mixing，且该 graph 与 AD-PSGD 共享。
2. Metropolis weights 行和为 1。
3. 同一 round 所有 client 使用同一 reference state。
4. 输出 test accuracy 和 virtual time。

### 17.2 AD-PSGD

1. 事件按 `next_finish_time` 升序执行。
2. 快 client 更新次数多于慢 client。
3. staleness 可被记录。
4. 无通信延迟时仍可因虚拟训练时间差产生缓存陈旧。

### 17.3 Clustered training

1. 每个 cluster 使用分簇输出的 members 和 leader。
2. 簇内按样本数加权聚合 client updates。
3. `leader_edges` 用于 Metropolis mixing。
4. `H` 控制簇间 mixing 频率。
5. global model 由 cluster models 样本数加权平均得到。

### 17.4 可复现性

同一 seed、同一配置下：

```text
metrics.csv
topology.json
final_summary.json
```

必须完全一致，允许浮点表示差异不超过固定容差。
