# 事故区多 CAV 协同控制建模说明

本文档以当前默认配置 `configs/env.yaml`、`configs/model.yaml`、`configs/train.yaml` 和
`configs/reward_candidate_A.yaml` 为准，作为代码、实验与论文之间的建模基线。

## 1. 系统边界与目标

- 道路：单向三车道，长度 $L=1200\,\mathrm{m}$，限速 $v_{\max}=30\,\mathrm{m/s}$。
- 事故：位于 $x_{acc}=700\,\mathrm{m}$，默认阻塞车道集合 $\mathcal B=\{0,1\}$，仅车道 2 可直接通行。
- 交通流：总需求 $4800\,\mathrm{veh/h}$，CAV 渗透率 $20\%$，其余车辆为 HDV。
- 控制对象：在途 CAV；HDV 由 SUMO 跟驰/换道模型控制。
- 控制目标：在保证动作可执行的前提下，提高事故近区速度和通行量，降低排队及阻塞车道滞留。

当前问题应建模为**动态智能体集合下的部分可观测协作马尔可夫博弈**，而不是固定车辆数的普通 MDP。

## 2. 马尔可夫博弈

定义

$$
\mathcal G=\langle\mathcal S,\{\mathcal O_i\},\{\mathcal A_i\},P,\{r_i\},\gamma\rangle.
$$

时刻 $t$ 的活跃 CAV 集合为 $\mathcal I_t$。仿真步长为 $\Delta t=0.1\,\mathrm{s}$，折扣因子
$\gamma=0.99$。环境状态 $s_t\in\mathcal S$ 包含道路内全部车辆的位置、速度、车道、类型以及事故状态；
Actor 不能直接访问 $s_t$，只接收局部观测 $o_i^t$。训练时中心化 Critic 使用固定槽位表示
$\tilde s_t=\operatorname{concat}(o_1^t,\ldots,o_N^t)$，其中 $N=4$，空槽位补零。

> 限制：$\tilde s_t$ 是“受控 CAV 观测拼接”，并非完整交通全局状态；因此代码中的 centralized critic
> 更准确地说是 centralized-observation critic。

## 3. 观测模型

每个 CAV 的观测为 17 维向量：

$$
o_i=[v_i,\ell_i,x_i,d_f,\Delta v_f,d_r,\Delta v_r,
 d_{lf},\Delta v_{lf},d_{lr},\Delta v_{lr},
 d_{rf},\Delta v_{rf},d_{rr},\Delta v_{rr},d_{acc},b_{mask}].
$$

其中同车道及左右相邻车道的前/后车信息在 $150\,\mathrm{m}$ 感知范围内提取。速度、位置、距离和相对速度均归一化；
缺失邻车用“最大距离、零相对速度”编码。$d_{acc}$ 是到事故点的有符号归一化距离；$b_{mask}$ 是阻塞车道位掩码的归一化值。

建模注意：位掩码用一个连续标量承载离散组合，网络可用但语义不够解耦。若扩展到更多车道或动态事故，建议改为逐车道 multi-hot 编码。

## 4. 动作与可行域

离散动作集合为

$$
\mathcal A_i=\{0:\mathrm{KEEP},1:\mathrm{LEFT},2:\mathrm{RIGHT}\}.
$$

车道 0 为最右侧，LEFT 增大车道索引。合法动作集合 $\mathcal A_i^{legal}(s_t)$ 由以下约束共同确定：

1. 目标车道索引存在；
2. 换道进行中禁止继续发出横向指令；
3. 冷却期可禁止与上次有效换道相反的方向。

Actor 输出 logits $z_i$ 后，在 softmax 前对非法动作置为大负数：

$$
\pi_\theta(a\mid o_i,m_i)=\operatorname{softmax}(z_i+(1-m_i)C),\quad C=-10^9.
$$

因此非法动作既不会被采样，也不会进入 PPO 的概率比计算。

## 5. Candidate A 奖励模型

个体速度收益：

$$R_i^{ind}=\operatorname{clip}(v_i/v_{max},0,1).$$

邻近 CAV 与局部混合交通速度分别为 $S_i^{cav}$、$S_i^{loc}$，群体收益为

$$R_i^{group}=\alpha S_i^{cav}+(1-\alpha)S_i^{loc},\qquad \alpha=0.6.$$

事故近区阻塞车道惩罚为

$$P_i^{block}=\mathbf 1(|x_i-x_{acc}|\le150)\,\mathbf 1(\ell_i\in\mathcal B).$$

默认总奖励：

$$
r_i=\operatorname{clip}(0.55R_i^{ind}+0.35R_i^{group}-0.10P_i^{block},0,1).
$$

该奖励是稠密速度奖励，能够训练，但仍有两项结构性偏差：

- 权重和为 $0.90$，阻塞惩罚又被最终截断，低速状态下惩罚区分度容易被压缩；
- 没有直接表达“完成绕行”“通行量”与“换道代价”，因此合法换道振荡仍可能是局部最优行为。

后续奖励改进应优先增加换道事件成本、方向反转成本和事故点通过事件奖励，并通过消融实验验证，避免仅凭单次 seed 调参。

## 6. MAPPO 模型

共享 Actor 与 Critic 均为两层 128 单元 ReLU MLP：

$$
\pi_\theta: \mathbb R^{17}\rightarrow\mathbb R^3,\qquad
V_\phi: \mathbb R^{68}\rightarrow\mathbb R.
$$

动态 CAV 通过 4 个固定槽位重绑定。空槽位在 rollout 中由 `agent_mask` 排除，GAE 在槽位失活处切断；
非终止 rollout 使用 $V(s_T)$ 自举。GAE 为

$$
\delta_t^i=r_t^i+\gamma V_\phi(s_{t+1})-V_\phi(s_t),\qquad
\hat A_t^i=\delta_t^i+\gamma\lambda\hat A_{t+1}^i,
$$

其中 $\lambda=0.95$。Actor 使用 PPO clipped objective，Critic 以同一时刻有效智能体 return 的均值作为团队价值目标。

这一实现等价于“共享团队 Critic + 个体奖励优势”。它可以工作，但不是严格的 per-agent centralized value。
若后续不同 CAV 的回报差异增大，建议令 Critic 输出 $N$ 个槽位价值，或输入待评估 agent 的 identity/局部观测后估计 $V_i(s,o_i)$。

## 7. 当前证据与结论边界

最新拥堵场景结果（`results/experiments/mappo_congested_v1/summary.json`，seed 42，180 s）显示：

- NoControl：平均速度 10.455 m/s，近区速度 5.712 m/s，通行量 80；
- RuleDivert：平均速度 11.278 m/s，近区速度 7.043 m/s，通行量 86；
- MAPPO：平均速度 10.737 m/s，近区速度 7.263 m/s，通行量 88。

因此当前模型在该 seed 上超过 NoControl，并在近区速度和通行量上略高于 RuleDivert，但平均速度与平均排队仍弱于 RuleDivert。
只有一个评估 seed，尚不能声称统计稳健优越。正式结论至少应使用 3–5 个独立 seed，报告均值、标准差与配对置信区间。

## 8. 下一阶段建模优先级

1. **先做多 seed 复评**：保持模型和奖励不变，验证当前收益是否稳定。
2. **显式抑制振荡**：启用并消融换道成本、反向换道成本；记录每车每公里换道次数。
3. **改进 Critic**：从单一团队标量升级为 agent-conditioned value，避免不同车辆 return 被均值目标过度平滑。
4. **扩充状态表达**：阻塞车道改为 multi-hot；增加距事故点的相对车道可达性和换道阶段状态。
5. **再考虑注意力**：只有在固定槽位 Critic 的多 seed 基线建立后，再引入邻居注意力，避免同时改变状态表示和训练目标。
