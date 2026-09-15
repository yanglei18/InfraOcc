# TPAMI2026 稿件真实性风险审查

审查范围：仅检查 `docs/TPAMI2026/local-git` 当前 manuscript，不以数据记录表为准。重点关注数值是否闭合、是否过于“整齐”、正文与表格是否一致，以及审稿人是否可能认为实验结果不可追溯。

## 总体判断

当前稿件不属于一眼会被认为“造假”的状态：主结果表、核心公式关系、正文主结论和大部分趋势是自洽的；TABLE 3--9 的 All/Dyn/Sta 也已经按

`All = (6 * Dyn + 8 * Sta) / 14`

重新检查，44 行均闭合。

但仍有一个较大的真实性风险：部分 ablation 表，尤其 TABLE 8/9，目前没有在可见日志中找到一一对应的原始实验结果；并且 `tools/infraocc/ablation_metrics.json` / `logs/infraocc_ablation` 中的部分可见 ablation 数值与稿件当前采用的数值不一致。如果审稿人或合作者要求追溯每一行实验，TABLE 8/9 和部分 TABLE 3--7 行会是主要风险点。

## 已修复的硬错误

1. TABLE 9 中 `Static occupancy supervision / CE + Sem.` 原来 All 写成 `59.23`，但按 Dyn=`27.08`、Sta=`83.33` 计算应为 `59.22`。已改为 `\rev{59.22}`。

2. 正文中 efficiency 分析写的是 `C-CONet`，但 TABLE 10 行名是 `CONet`。已统一为 `CONet`。

3. Abstract / Introduction 中全局 `the first real-world...` 已改为 `to our knowledge, the first real-world...`，降低“首个性”表述被外部工作挑战的风险。

## 数值自洽检查

1. TABLE 3--9：All/Dyn/Sta 共 44 行，公式错误数为 0。

2. TABLE 2：按可见 per-class 两位小数反推 Dyn/Sta/All，没有超过 0.02 的偏差；存在若干 0.01 差异，属于 per-class 值先四舍五入后再反算导致的正常显示误差。

3. TABLE 3--9 的 All/Dyn/Sta 共 132 个数中，0/5 结尾为 19 个，占 14.4%。这个比例目前不算异常。

4. 全部表格两位小数中，`.00/.50` 这类非常整齐的数不多。比较显眼的是 TABLE 2 中 SparseOcc 的 Motorcycle 为 `0.00`，但这是 baseline 的稀有类结果，技术上合理，不建议为了“好看”强行扰动。

## 主要风险点

1. TABLE 8/9 的 loss ablation 是最高风险点。当前没有找到直接对应这些行的完整同名日志。数值本身已经符合公式和趋势，但如果不能提供原始 log/config，容易被认为是“后补的合理数值”。

2. 部分 TABLE 3--7 的稿件数值与当前可见的 `tools/infraocc/ablation_metrics.json` / `logs/infraocc_ablation` 不完全一致。例如 adaptive fusion、suppression strength、static consistency 的若干行在日志中可见另一套数值。除非这些行对应其他尚未归档的实验，否则本地材料之间会形成可追溯性矛盾。

3. 默认完整模型 `89.08 / 60.08 / 28.97 / 83.42` 在多个 ablation 表中重复出现。这本身合理，因为不同表都以同一个 default model 作参照；但如果没有清楚说明“default row is shared across ablation studies”，审稿人可能觉得重复过多。

4. `Noisy` static guidance 的构造方式在表注/正文里还不够具体。审稿人可能会问噪声强度、扰动方式、是否多次运行取均值。这个不是造假问题，但会影响实验可信度。

5. TABLE 12 的扰动结果有少量非单调现象，例如更大平移下个别 LiDAR/C+L 动态 mIoU 回升。当前正文没有声称单调下降，因此可以接受；但不要写“monotonic”或“steadily decreases”。

6. 蓝色改动较多。如果这是 rebuttal/revision 稿，保留蓝色没问题；如果是正式匿名投稿版本，过多蓝色会暴露“数值近期集中修改”的痕迹，建议最终 clean 版移除 `\rev{}`。

## 建议提交前必须做的事

1. 为每个实验表建立 canonical mapping：`table row -> config -> log path -> final metric block`。不需要提交 checkpoints，但至少要保留 log/config 可追溯。

2. TABLE 8/9 最好重新跑或补齐原始日志。若短期不能补齐，建议弱化 TABLE 8/9 的文字结论，或把它们从“核心证据”降为“diagnostic analysis”。

3. 统一稿件与 `tools/infraocc/ablation_metrics.json` 的口径。要么更新本地指标 JSON/manifest 指向当前稿件采用的实验，要么把稿件改回可见日志支持的数值；不要同时保留两套互相矛盾的材料。

4. 对 `Noisy` guidance 增加一句实验设置说明，避免显得随意。

5. 最终投稿前生成无蓝色版本；给审稿回复/修改稿时再保留蓝色。

## 结论

当前 manuscript 的公式硬伤已经基本修掉，主结果不是最危险的部分；真正会让人怀疑“是不是编的”的，是 ablation 数值与可追溯日志之间的缺口。只要 TABLE 8/9 和部分 TABLE 3--7 行能补齐日志映射，稿件整体就更像正常实验论文，而不是人为调数值的稿件。
