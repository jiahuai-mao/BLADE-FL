# 分簇算法需求文档

## 1. 目标

本模块实现论文 `main260716.tex` 中 `Balanced topology construction`、`Dynamic membership and topology refresh`，以及 `Supplemental Material260716.tex` 中 `Supplementary Method 2: Deterministic topology construction`、`Supplementary Method 3: Dynamic topology refresh` 描述的分簇与拓扑维护算法。

第一阶段只实现固定 `K` 下的 deterministic clustering，不实现聚合算法，不引入通信时间或带宽。分簇结果作为后续聚合算法的输入拓扑。

核心目标：

1. 将 active clients 分成 `K` 个 cluster。
2. 每个 cluster 的 client 数量满足论文容量硬约束。
3. 在容量硬约束下最小化平衡分簇目标 `J_bal`。
4. 分簇后选择 leader，并保证 leader overlay 连通。
5. 在拓扑刷新时维护 active clients、feasible communication graph、partition 和 leader overlay。
6. 输出可复现的 `Topology`，供后续聚合算法使用。

## 2. 非目标

第一阶段不做以下内容：

1. 不实现 FedAvg、DPSGD、ADPSGD 或其他聚合算法。
2. 不把通信时间、带宽、上传/下载延迟加入分簇 score。
3. 不使用运行时 wall-clock 作为真实客户端耗时。
4. 第一阶段代码可暂不实现动态 join/leave repair，但需求文档必须保留 Supplementary Method 3 的完整实现要求。
5. 不承诺求解全局最优 partition，只实现论文中的 deterministic feasible construction 和 deterministic repair/reconstruction。

## 3. 输入

每个 active client 提供一个 descriptor：

```text
(i, T_i^loc, n_i, p_i)
```

含义：

1. `i`: client identifier。
2. `T_i^loc`: 当前 client 完成固定 local workload 的估计训练时间。
3. `n_i`: client 本地样本数。
4. `p_i`: client 的归一化类别直方图。

在监督分类任务中：

```text
p_i^c = n_i^c / n_i
```

其中 `n_i^c` 是 client `i` 上类别 `c` 的样本数。

在当前单设备模拟中，`T_i^loc` 第一版按数据量构造：

```text
T_i^loc = n_i
```

如后续需要考虑 local epochs，可扩展为：

```text
T_i^loc = local_epochs * n_i
```

注意：`T_i^loc` 不包含模型传输时间，也不包含 leader 等待时间。

## 4. 输出

分簇模块输出：

```python
Topology(
    clusters: list[list[int]],
    leaders: list[int],
    leader_edges: list[tuple[int, int]],
    client_to_cluster: dict[int, int],
)
```

同时输出用于审计和实验记录的 metrics：

```python
{
    "K": int,
    "capacities": list[int],
    "waiting_cost": float,
    "distribution_cost": float,
    "normalized_waiting_cost": float,
    "normalized_distribution_cost": float,
    "balanced_score": float,
    "leader_score": float,
    "leader_edges": list[tuple[int, int]],
    "connected": bool,
}
```

## 5. 硬约束：cluster capacity

对固定候选 cluster 数 `K`，令：

```text
N^s = number of active clients
r = N^s - K * floor(N^s / K)
```

容量分配为：

```text
capacity[k] = ceil(N^s / K),  for k < r
capacity[k] = floor(N^s / K), for k >= r
```

即前 `r` 个 cluster 的容量为 `ceil(N^s / K)`，其余 cluster 的容量为 `floor(N^s / K)`。

该约束是硬约束，不进入 score。初始化、插入、swap 后均必须满足固定容量。

## 6. 分簇目标函数

### 6.1 Waiting cost

论文中的等待代价为：

```text
J_T =
sum_k sum_{C_i in V_k^s}
(
max_{C_j in V_k^s} T_j^loc - T_i^loc
)
```

含义：簇内同步时，所有 client 等待最慢 client 造成的 idle time。

实现函数：

```python
def waiting_cost(clusters, clients) -> float:
    ...
```

### 6.2 Distribution cost

全局 sample-weighted class histogram：

```text
p_G^s = sum_i n_i p_i / sum_i n_i
```

cluster `k` 的 sample-weighted class histogram：

```text
p_k^s =
sum_{C_i in V_k^s} n_i p_i
/
sum_{C_i in V_k^s} n_i
```

分布代价：

```text
J_D =
1 / K * sum_k || p_k^s - p_G^s ||_1
```

实现函数：

```python
def distribution_cost(clusters, clients, global_hist) -> float:
    ...
```

### 6.3 Normalisation

补充材料中的归一化规则：

```text
Jbar_T =
J_T / (N^s * (T_max^s - T_min^s))

Jbar_D =
J_D / 2
```

其中：

```text
T_max^s = max_i T_i^loc
T_min^s = min_i T_i^loc
```

### 6.4 Balanced clustering score

固定 `K` 下的分簇 score：

```text
J_bal = alpha_T * Jbar_T + alpha_D * Jbar_D
alpha_T + alpha_D = 1
```

实现中禁止引入未在论文中定义的 `time_score = max / mean` 或 `time_weighted_load`。插入和 swap 只比较：

```text
delta = J_bal(proposal) - J_bal(current)
```

## 7. Deterministic partition construction

### 7.1 Client ordering

先计算每个 client 与全局类别分布的距离：

```text
d_i = || p_i - p_G^s ||_1
```

client 排序规则严格为：

```text
1. decreasing d_i
2. then decreasing T_i^loc
3. then increasing client identifier
```

实现中必须保证同一输入下排序稳定、结果可复现。

### 7.2 Seed assignment

排序后的前 `K` 个 client 分别放入不同 cluster：

```text
ordered_clients[0] -> cluster 0
ordered_clients[1] -> cluster 1
...
ordered_clients[K-1] -> cluster K-1
```

若某个 cluster 容量为 0，则该 `K` 不可行。正常候选 `K` 应保证每个 cluster 容量至少为 1。

### 7.3 Greedy insertion

对剩余 client，按排序顺序逐个插入。

候选 cluster 必须满足：

```text
len(cluster[k]) < capacity[k]
```

选择规则：

```text
选择使 J_bal 增量最小的 cluster
若多个 cluster 增量相同，选择 cluster index 更小者
```

实现函数建议：

```python
def assign_remaining_clients(
    clusters,
    remaining_clients,
    capacities,
    clients,
    params,
) -> list[list[int]]:
    ...
```

### 7.4 Exchange-based refinement

初始分配后进行 swap refinement。

只考虑不同 cluster 中两个 client 的交换。由于交换不改变 cluster size，因此天然保持固定容量。

每一步：

1. 枚举所有 eligible swaps。
2. 计算 `delta = J_bal(after_swap) - J_bal(current)`。
3. 找到下降最大的 swap。
4. 只有当下降量超过 `epsilon_swap` 时才接受：

```text
current_score - proposal_score > epsilon_swap
```

停止条件：

```text
无可接受 swap
或 accepted swaps 达到 R_swap
```

tie-break 规则：

```text
按 cluster index 和 client identifier 升序决定
```

建议实现时枚举顺序固定为：

```python
for k1 in range(K):
    for k2 in range(k1 + 1, K):
        for cid1 in sorted(clusters[k1]):
            for cid2 in sorted(clusters[k2]):
                ...
```

这样天然满足确定性 tie-break。

## 8. Leader selection

分簇完成后选择 leader。

每个 cluster 内 leader 候选排序：

```text
1. increasing T_i^loc
2. then increasing client identifier
```

需要从每个 cluster 选择一个 leader，使 selected leaders 在 feasible communication graph 上诱导出的 leader overlay 连通。

leader overlay 必须显式输出为无向边列表：

```text
E_L^s(K) = induced_edges(selected_leaders, feasible_graph)
```

代码中表示为：

```python
leader_edges: list[tuple[int, int]]
```

每条边必须使用规范化顺序：

```text
(min(leader_i, leader_j), max(leader_i, leader_j))
```

整个 `leader_edges` 列表按升序排序，保证同一输入下输出完全确定。

如果没有真实 feasible communication graph，第一阶段默认使用 complete graph。此时 selected leaders 两两相连：

```text
|E_L^s(K)| = K(K - 1) / 2
```

该默认行为必须写入输出 metrics 或运行配置，避免把 complete overlay 误解为真实网络测量结果。

对每个 connected leader combination，leader score 为：

```text
J_L = sum_k T_{ell_k}^loc
```

选择规则：

```text
1. 最小 J_L
2. 若 J_L 相同，选择 leader identifier list 字典序更小者
```

如果不存在 connected leader combination，则该 partition/topology 不可行。

第一阶段如果没有真实 feasible communication graph，可使用 complete graph 作为默认图。该默认必须在配置和日志中明确记录。

## 9. 最佳分簇数量 K 的选择

### 9.1 当前实现阶段

第一阶段已经实现固定 `K`：

```yaml
clustering:
  k: 2
```

固定 `K` 时，只构造该 `K` 对应的 partition、leader 和 topology，不比较不同 `K`。

### 9.2 后续目标：K=auto

下一阶段需要实现论文中的候选 `K` 搜索。给定 active-client 数量：

```text
N^s = |C^s|
```

候选集合为：

```text
K in {K_min, ..., floor(N^s / n_cluster_min)}
```

其中：

1. `K_min` 是允许的最小 cluster 数量。
2. `n_cluster_min` 是每个 cluster 允许的最小 client 数量。
3. 候选 `K` 必须满足 `1 <= K <= N^s`。
4. 若候选集合为空，则应报错并说明当前 active-client 数不足。

配置建议：

```yaml
clustering:
  k: auto
  k_min: 2
  min_clients_per_cluster: 2
```

### 9.3 每个候选 K 的处理流程

对每个候选 `K`，执行完整 deterministic topology construction：

1. 按 `K` 计算固定 cluster capacities。
2. 按 `d_i`、`T_i^loc`、client id 排序。
3. seed assignment。
4. greedy insertion。
5. exchange-based refinement。
6. connectivity-aware leader selection。
7. 若无法选择 connected leader overlay，则该 `K` 不可行。
8. 记录该 `K` 的 `J_bal`、leader set、leader overlay edge set 和 communication proxy。

注意：不同 `K` 的 partition 都必须独立构造，不应复用另一个 `K` 的分簇结果。

### 9.4 Communication proxy

论文中用于候选 topology 比较的通信代理为：

```text
B_comm^s(K)
=
2B(N^s - K)
+
2B |E_L^s(K)| / H
```

其中：

1. `B` 是一次传输的模型或模型更新大小。
2. `E_L^s(K)` 是该候选 `K` 下 selected leader overlay 的边集合。
3. `H` 是 inter-cluster mixing interval。
4. 第一项 `2B(N^s-K)` 表示标准化周期内每个 non-leader client 的一次模型下载和一次模型更新上传。
5. 第二项 `2B|E_L^s(K)|/H` 表示 leader-overlay 边上的双向传输，并按 mixing interval `H` 摊销。

该 proxy 只用于比较候选 topologies，不表示真实异步执行中的精确通信量。

当前用户要求“不考虑通信时间和带宽”，这与该 proxy 不冲突：这里的 `B_comm` 是传输字节量 proxy，不是通信耗时，也不需要带宽参数。

实现函数建议：

```python
def communication_proxy(
    num_clients: int,
    k: int,
    leader_edges: list[tuple[int, int]],
    model_size: float,
    mixing_interval: int,
) -> float:
    ...
```

如果只关心相对比较，且 `B` 对所有候选相同，可以设置：

```text
B = 1
```

此时：

```text
B_comm^s(K)
=
2(N^s - K)
+
2|E_L^s(K)| / H
```

### 9.5 Communication proxy 归一化

对所有 feasible candidates，先得到：

```text
B_min^s = min_K B_comm^s(K)
B_max^s = max_K B_comm^s(K)
```

归一化：

```text
Jbar_C(K)
=
(
B_comm^s(K) - B_min^s
)
/
(
B_max^s - B_min^s 
)
```

如果所有 feasible candidates 的 communication proxy 相同，则：

```text
Jbar_C(K) = 0
```

### 9.6 Topology score

自动选择 `K` 时，论文使用：

```text
J_K = (1 - lambda_C) * J_bal + lambda_C * Jbar_C(K)
```

其中：

1. `J_bal` 是该候选 `K` 下分簇完成后的 balanced clustering score。
2. `Jbar_C(K)` 是归一化 communication proxy。
3. `lambda_C` 控制 topology balance 和通信 proxy 的权重。
4. `0 <= lambda_C <= 1`。

配置建议：

```yaml
clustering:
  lambda_communication: 0.2
  model_size: 1.0
  mixing_interval: 5
```

如果当前实验阶段暂时不希望通信 proxy 影响 `K`，可以设置：

```text
lambda_C = 0
```

此时：

```text
J_K = J_bal
```

### 9.7 最佳 K 选择规则

在所有 feasible candidates 中：

```text
K_best = argmin_K J_K
```

tie-break：

```text
若多个 K 的 J_K 相同，选择更小的 K
```

如果某个 `K` 无法形成 connected leader overlay，则该候选被丢弃。

如果所有候选 `K` 都不可行，则返回错误：

```text
No feasible topology found for active clients.
```

### 9.8 输出要求

启用 `K=auto` 时，除了最终 `Topology`，还应输出候选搜索记录：

```python
{
    "selected_k": int,
    "candidate_rows": [
        {
            "K": int,
            "feasible": bool,
            "capacities": list[int],
            "balanced_score": float,
            "waiting_cost": float,
            "distribution_cost": float,
            "normalized_waiting_cost": float,
            "normalized_distribution_cost": float,
            "leader_score": float | None,
            "leader_edges": list[tuple[int, int]],
            "communication_proxy": float | None,
            "normalized_communication_proxy": float | None,
            "topology_score": float | None,
            "selected": bool,
            "failure_reason": str | None,
        }
    ]
}
```

落盘建议：

```text
results/cluster_only/{run_name}/
  topology.json
  cluster_metrics.json
  k_search.json
```

### 9.9 代码接口建议

在 `clustering.py` 中增加：

```python
def candidate_k_values(
    num_clients: int,
    k_min: int,
    min_clients_per_cluster: int,
) -> list[int]:
    ...

def select_best_k_topology(
    clients: list[ClientMetadata],
    k_min: int,
    min_clients_per_cluster: int,
    alpha_time: float,
    alpha_distribution: float,
    lambda_communication: float,
    model_size: float,
    mixing_interval: int,
    feasible_graph: Graph,
    epsilon: float,
    epsilon_swap: float,
    max_swaps: int,
) -> tuple[Topology, dict]:
    ...
```

`build_deterministic_balanced_clusters(...)` 继续负责固定 `K`，`select_best_k_topology(...)` 负责枚举候选 `K` 并调用固定 `K` 构造函数。

### 9.10 验收标准

1. 候选 `K` 集合严格等于 `{K_min, ..., floor(N^s / n_cluster_min)}`。
2. 每个候选 `K` 独立执行 deterministic topology construction。
3. 不可形成 connected leader overlay 的候选被标记为 infeasible。
4. `B_comm^s(K)` 与论文公式一致。
5. `Jbar_C(K)` 只在 feasible candidates 内归一化。
6. `J_K` 与论文公式一致。
7. 平局时选择更小 `K`。
8. 同一输入重复运行，`K_best` 和候选记录完全一致。

## 10. Dynamic topology refresh

本节对应 `main260716.tex` 中 `Dynamic membership and topology refresh` 和 `Supplemental Material260716.tex` 中 `Supplementary Method 3: Dynamic topology refresh`。

### 10.1 固定拓扑区间

在一个 fixed-topology interval 内：

1. cluster membership 保持不变。
2. leader overlay 保持不变。
3. 临时 client unavailability 只影响当前 cluster update 的 participating set。
4. 某个 client 没有参与一次 update，不应立刻从所属 cluster 移除。
5. 只有在 topology refresh 时才处理 persistent membership changes 和 persistent link changes。

### 10.2 刷新输入

一次 topology refresh 至少需要以下输入：

```python
previous_topology: Topology | None
previous_cluster_models: list[ModelVector] | None
previous_active_clients: set[int]
updated_client_metadata: list[ClientMetadata]
departed_clients: set[int]
newly_joined_clients: set[int]
updated_feasible_graph: Graph
clustering_params: dict
```

其中：

1. `updated_client_metadata` 必须包含刷新后所有 active clients 的 descriptor。
2. newly joined clients 必须在收集 descriptor 后才可进入 clustering。
3. departed clients 或 persistent unavailable clients 必须从 active-client set 中移除。
4. persistent link changes 必须先应用到 feasible communication graph，再进行 repair 或 reconstruction。

### 10.3 刷新流程总览

每次 topology refresh 的处理顺序必须为：

1. 更新 active-client set。
2. 更新 feasible communication graph。
3. 检查上一轮 cluster 数 `K^s` 对新的 active-client 数是否仍可行。
4. 若 `K^s` 可行，优先尝试 repair previous partition。
5. repair 成功时保留 repaired topology。
6. 若 `K^s` 不可行，或 repair 无法同时恢复 capacity 和 connected leader overlay，则重新运行完整 deterministic topology construction。
7. reconstruction 可使用不同的 cluster 数 `K^{s+1}`。
8. 如果 partition 改变，则按 membership overlap warm-start 新 cluster models。

### 10.4 `K^s` 可行性检查

令刷新后的 active-client 数为：

```text
N^{s+1} = |C^{s+1}|
```

上一轮 cluster 数 `K^s` 仍可用于 repair 的条件为：

```text
K_min <= K^s <= floor(N^{s+1} / n_cluster_min)
```

并且：

```text
1 <= K^s <= N^{s+1}
```

如果上述条件不满足，不得尝试 repair，必须直接执行完整 reconstruction。

### 10.5 Partition repair

当 `K^s` 仍可行时，先尝试修复上一轮 partition。

#### 10.5.1 移除 departed clients

先从其原 cluster 中移除：

```text
departed clients
persistent unavailable clients
```

移除后必须重新计算目标 capacities：

```text
capacity[k] = ceil(N^{s+1} / K^s),  for k < r
capacity[k] = floor(N^{s+1} / K^s), for k >= r
```

容量规则必须与 Supplementary Method 2 完全相同。

#### 10.5.2 插入 newly joined clients

newly joined clients 在 descriptor 已收集后，按确定性顺序插入可用 cluster。

插入候选 cluster 必须满足：

```text
len(cluster[k]) < capacity[k]
```

选择规则：

```text
选择使 J_bal 增量最小的 cluster
若多个 cluster 增量相同，按 cluster index 升序选择
```

`J_bal`、`Jbar_T`、`Jbar_D` 的定义必须与固定 `K` 构造阶段完全一致，不得加入额外 repair score。

#### 10.5.3 修复 oversized clusters

如果有 cluster 超过目标 capacity，则需要从 oversized clusters 向 underfilled clusters 移动 clients。

一次 move 必须满足：

```text
source cluster is oversized
target cluster is underfilled
move reduces capacity violation
```

在所有可行 moves 中选择：

```text
使 J_bal 增量最小的 move
```

tie-break：

```text
按 source cluster index、target cluster index、client identifier 升序决定
```

重复执行 moves，直到所有 cluster 满足目标 capacities；如果无法恢复 capacity，则 repair 失败。

#### 10.5.4 Exchange-based refinement

一旦 target capacities 恢复，必须应用 Supplementary Method 2 中相同的 exchange-based refinement：

```text
枚举不同 clusters 中两个 clients 的交换
接受使 J_bal 下降最大的 exchange
下降量必须超过 epsilon_swap
停止于无可接受 exchange 或达到 R_swap
```

该 refinement 不得破坏固定 capacities。

### 10.6 Refresh 后 leader selection

repair 或 reconstruction 得到 partition 后，必须重新执行 connectivity-aware leader selection：

1. 每个 cluster 内 leader 候选按 `T_i^loc` 升序、client id 升序排序。
2. 从每个 cluster 选择一个 leader。
3. selected leaders 在 updated feasible graph 上诱导出的 leader overlay 必须 connected。
4. connected combinations 中选择 `J_L = sum_k T_{ell_k}^loc` 最小者。
5. `J_L` 相同则选择 leader identifier list 字典序更小者。

repaired topology 只有同时满足以下条件时才可保留：

```text
target capacities restored
connected leader overlay formed
```

否则 repair 失败，必须执行完整 reconstruction。

### 10.7 Reconstruction fallback

当以下任一条件成立时，必须重新运行完整 deterministic topology construction：

1. previous `K^s` 对新的 active-client 数不再可行。
2. repair 无法恢复 target capacities。
3. repair 后无法形成 connected leader overlay。
4. updated feasible graph 导致所有 repaired leader choices 不可连通。

reconstruction 流程必须与 `K=auto` 完全一致：

1. 枚举候选 `K in {K_min, ..., floor(N^{s+1}/n_cluster_min)}`。
2. 对每个候选 `K` 独立执行 deterministic topology construction。
3. 丢弃无法形成 connected leader overlay 的候选。
4. 用 `J_K = (1 - lambda_C) * J_bal + lambda_C * Jbar_C(K)` 选择最终 topology。
5. 平局时选择更小 `K`。

reconstructed topology 可以使用不同的 cluster 数。

### 10.8 无 connected leader overlay 时的运行规则

如果在 topology refresh 时没有任何 connected leader overlay 可构造：

1. 有 available leader 的 clusters 可以继续独立执行 local updates。
2. inter-cluster mixing 必须暂停。
3. 没有 available leader 的 cluster 必须暂停 local updates。
4. 不得继续使用已经失效的 leader overlay。
5. 下一次 topology refresh 必须再次尝试 topology construction。
6. 一旦 connected overlay 恢复，neighbouring leaders 必须交换当前 cluster models，用于初始化 neighbour caches，然后才能恢复 inter-cluster mixing。

### 10.9 Repartitioning 后的 model transfer

当 partition 改变时，新 cluster 不应简单从随机初始模型开始，而应根据上一轮 cluster models warm-start。

对新 cluster `V_k^{s+1}`，定义保留客户端集合：

```text
R_k^{s+1} = V_k^{s+1} ∩ C^s
```

如果：

```text
R_k^{s+1} != empty
```

则上一轮 cluster `V_j^s` 对新 cluster `V_k^{s+1}` 的 transfer weight 为：

```text
beta_{kj}^{s+1}
=
sum_{C_i in V_k^{s+1} ∩ V_j^s} n_i
/
sum_{C_i in R_k^{s+1}} n_i
```

新 cluster model 初始化为：

```text
m_k^{s+1,init}
=
sum_{j=1}^{K^s} beta_{kj}^{s+1} m_j^{s,last}
```

含义：上一轮 cluster 对新 cluster 的贡献，与它在新 cluster 中保留下来的样本量成比例。

如果新 cluster 只包含 newly joined clients，即：

```text
R_k^{s+1} = empty
```

则新 cluster model 初始化为上一轮 cluster models 的均值：

```text
m_k^{s+1,init}
=
1 / K^s * sum_{j=1}^{K^s} m_j^{s,last}
```

第一轮 clustering 时，所有 clusters 从相同 initial model 开始。

### 10.10 Dynamic refresh 输出

每次 refresh 应输出审计记录：

```python
{
    "refresh_round": int,
    "previous_k": int | None,
    "new_k": int | None,
    "repair_attempted": bool,
    "repair_succeeded": bool,
    "reconstruction_used": bool,
    "departed_clients": list[int],
    "newly_joined_clients": list[int],
    "removed_unavailable_clients": list[int],
    "capacity_violation_before_repair": dict,
    "capacity_violation_after_repair": dict,
    "connected_overlay_available": bool,
    "mixing_suspended": bool,
    "paused_clusters": list[int],
    "leader_edges": list[tuple[int, int]],
    "model_transfer_weights": dict,
    "failure_reason": str | None,
}
```

建议落盘：

```text
results/.../{run_name}/
  topology_refresh_log.json
```

### 10.11 Dynamic refresh 接口建议

建议新增：

```python
def refresh_topology(
    previous_topology: Topology | None,
    previous_clients: list[ClientMetadata],
    updated_clients: list[ClientMetadata],
    previous_cluster_models: list[Tensor] | None,
    updated_feasible_graph: Graph,
    params: ClusteringParams,
) -> tuple[Topology | None, dict]:
    ...
```

其中 `Topology | None` 表示：当没有 connected leader overlay 可构造时，返回 `None` 或返回一个带 `connected=False` 状态的 refresh result；训练模块必须据此暂停 inter-cluster mixing。

### 10.12 Dynamic refresh 验收标准

1. refresh 前先更新 active-client set 和 feasible graph。
2. previous `K^s` 可行时，必须先尝试 repair，再 reconstruction。
3. departed clients 必须从原 cluster 移除。
4. newly joined clients 必须在 descriptor 收集后插入有容量 cluster。
5. oversized clusters 必须通过 capacity-reducing moves 修复。
6. repair 后必须应用 exchange-based refinement。
7. repaired topology 只有同时满足 capacity 和 connected overlay 时才保留。
8. repair 失败或 `K^s` 不可行时，必须 rerun full topology construction。
9. reconstruction 可以选择不同 `K`。
10. 无 connected overlay 时，inter-cluster mixing 必须暂停。
11. 无 available leader 的 cluster 必须暂停 local updates。
12. partition 改变时，cluster models 必须按 membership overlap warm-start。
13. 同一 refresh 输入和 seed 下，repair/reconstruction 结果必须完全确定。

## 11. 配置项

建议配置：

```yaml
clustering:
  method: deterministic_balanced
  k: 4
  alpha_time: 0.5
  alpha_distribution: 0.5
  epsilon_swap: 1.0e-12
  max_swaps: 10
  time_mode: num_samples
  leader_graph: complete
  dynamic_refresh:
    enabled: false
    refresh_interval: 100
    repair_before_reconstruction: true
    pause_mixing_without_connected_overlay: true
```

约束：

```text
alpha_time + alpha_distribution = 1
```

## 12. 建议代码结构

```text
clustered_dfl/
  types.py
  data.py
  clustering.py
  topology.py
  metrics.py
scripts/
  run_cluster_only.py
```

### 12.1 types.py

```python
@dataclass
class ClientMetadata:
    client_id: int
    num_samples: int
    label_counts: list[int]
    train_time: float

@dataclass
class Topology:
    clusters: list[list[int]]
    leaders: list[int]
    leader_edges: list[tuple[int, int]]
    client_to_cluster: dict[int, int]
```

### 12.2 clustering.py

核心函数：

```python
def build_deterministic_balanced_clusters(
    clients: list[ClientMetadata],
    k: int,
    alpha_time: float,
    alpha_distribution: float,
    epsilon_swap: float,
    max_swaps: int,
    leader_graph: Graph,
) -> tuple[Topology, dict]:
    ...
```

内部函数：

```python
compute_client_histograms(...)
compute_global_histogram(...)
compute_capacities(...)
waiting_cost(...)
distribution_cost(...)
balanced_score(...)
order_clients_for_seeding(...)
greedy_insert(...)
refine_by_swaps(...)
select_connected_leaders(...)
induced_leader_edges(...)
```

## 13. 验收标准

### 13.1 正确性

1. 所有 client 恰好出现一次。
2. 每个 cluster size 等于预先分配的 capacity。
3. 输出 leader 数量等于 `K`。
4. `leader_edges` 等于 selected leaders 在 feasible communication graph 上诱导出的无向边集合。
5. `client_to_cluster` 与 `clusters` 完全一致。
6. 若使用 complete graph，则 leader overlay 必然 connected。
7. `J_T/J_D/J_bal` 与公式一致。

### 13.2 确定性

同一输入、多次运行，输出必须完全一致：

```text
clusters
leaders
metrics
```

### 13.3 论文一致性

代码中不得出现以下替代指标作为分簇目标：

```text
cluster_size_balance_score
max(cluster_time) / mean(cluster_time)
time_weighted_load
bandwidth cost
communication time
```

允许出现的目标项只有：

```text
Jbar_T
Jbar_D
J_bal
```

## 14. 测试计划

1. Unit test: capacity 分配。
2. Unit test: `p_i`、`p_G`、`p_k` 计算。
3. Unit test: `J_T` 计算。
4. Unit test: `J_D` 计算。
5. Unit test: client ordering tie-break。
6. Unit test: greedy insertion 不超过 capacity。
7. Unit test: swap refinement 只接受下降超过 `epsilon_swap` 的交换。
8. Unit test: leader selection 选择最小 `J_L` 的 connected combination。
9. Integration test: CIFAR10 Dirichlet 20 clients, fixed `K=2` 输出 topology。
10. Unit test: dynamic refresh 移除 departed clients。
11. Unit test: dynamic refresh 插入 newly joined clients 且不超过 capacity。
12. Unit test: oversized clusters 通过最小 `J_bal` 增量 moves 修复。
13. Unit test: repair 后无法形成 connected overlay 时触发 reconstruction。
14. Unit test: 无 connected overlay 时 mixing_suspended=True。
15. Unit test: repartitioning 后 model transfer weights 与 overlap 样本数一致。

## 15. 后续扩展

后续可以在该分簇模块基础上继续实现：

1. Dynamic topology refresh 代码实现。
2. Membership repair 代码实现。
3. 聚合算法。
4. 异步 inter-cluster mixing。

这些扩展必须保持分簇模块接口稳定：聚合算法只消费 `Topology`，不依赖分簇内部实现细节。
