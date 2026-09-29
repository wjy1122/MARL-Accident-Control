# MARL-Accident-Control-V2

**基于群体收益与合法动作约束的多智能体强化学习高速公路事故区多 CAV 协同换道控制**

English (working title): *MAPPO with Group Benefit for Cooperative CAV Lane-Changing in Highway Accident Zones*

> 论文草稿见 `paper/`。当前实现以 **MAPPO + Group Benefit + 动作掩码/信用分配修正** 为主；邻居注意力为后续扩展，不作为已完成贡献。

建模口径、数学定义、当前证据与下一阶段优先级见 [MODELING.md](MODELING.md)。

## 研究问题

高速公路混合交通（HDV + CAV）在事故导致部分车道阻塞时，多辆联网自动驾驶车辆如何协同决策（保持 / 换道），以改善事故区交通运行效率，并避免仅优化个体利益带来的相互干扰。

## 研究场景

```text
高速公路 → 3车道混合交通 → HDV + CAV → 交通事故 / 车道阻塞
→ 多辆 CAV 接近事故区 → 协同决策 → 改善通行效率
```

## 核心方法（与论文对齐）

- **MAPPO**（CTDE）：参数共享 Actor + 中心化 Critic
- **Group Benefit / Candidate A**：个体速度收益 + 群体/局部通行效率 + 近区阻塞车道惩罚
- **合法动作掩码 + 信用分配修正**：消除非法换道梯度污染，修正空槽 GAE / 末端自举 / 换道中指令冲突

最终方法名：**MAPPO with Group Benefit**（可选扩展：Neighbor Attention）

## 场景版本

| 配置 | 说明 |
|------|------|
| `configs/env.yaml` | **Congested v1**：封闭车道 `[0,1]`，需求 `4800` veh/h（当前默认） |
| `configs/env_legacy_1lane_3600.yaml` | 旧正式实验：封闭 `[0]`，`3600` veh/h |

改完 `env.yaml` 后需重建 SUMO：

```text
.venv\Scripts\python.exe scripts\build_sumo_scenario.py
```

拥堵验收（NoControl vs 规则绕行）：

```text
.venv\Scripts\python.exe scripts\run_congested_scenario_baselines.py --seconds 180
```

在拥堵场景上正式训练 + 评估（Candidate A + Mask，约 100 updates）：

```text
.venv\Scripts\python.exe scripts\run_mappo_congested_v1.py
```

## 开发阶段

| Phase | 内容 | 状态 |
|-------|------|------|
| Phase 1 | SUMO 场景 | 完成 |
| Phase 2 | Environment / Reward | 完成（含 Candidate A） |
| Phase 3 | Observation / Action mask | 完成 |
| Phase 4 | MAPPO 训练与评估管线 | 完成 |
| Phase 5 | 掩码与信用分配消融 | 进行中 / 部分完成 |
| Phase 6 | 打赢 No Control 主指标 + 规则基线 | 待完成（投稿关键） |
| Phase 7 | Neighbor Attention（可选） | 未开始 |
| Phase 8 | 多 seed / 渗透率敏感性 / 论文定稿 | 待完成 |

## 环境

优先使用项目虚拟环境：

```text
.venv\Scripts\python.exe
```

不要使用系统 Python 或 Conda 作为项目解释器。

## 说明

本仓库为 V2 重写版。旧项目 `MARL-Accident-Control` 仅作参考，不直接打补丁。

