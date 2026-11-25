# SOM–LLM–UCB 双链工作流

双态分子矩阵进入 SOM。SOM 把整体结构和局部化学单元映射到同一结构空间。格内候选依据结构覆盖、局部差异和卡尔曼评分映射送评。结构差分和整体、局部评分映射为 LLM 定向母体设计提供输入。种群扩展采用 UCB 分配的杂交、突变和外源补入。

程序检索整体 Memory 得到类比分，按位点匹配局部证据。LLM 判读局部证据的效应方向，程序结合整体类比分和局部判断。送评队列全部取得有效评分后才提交该代，并将反馈写入评分映射和 UCB。该版配置最多运行 10 代、每代最多送评 8 个分子、总计最多 100 次 API 调用。

| 模块 | 路径 |
| --- | --- |
| 实验入口 | `run_dual_ten_generation.py` |
| 化学单元与 SOM | `SOM/chemical_unit_som_v1/` |
| 卡尔曼映射和矩阵编辑接口 | `LLM/frontier_loop/` |
| Memory 匹配与评分 | `LLM/overall_local_v1/` |
| 评分协议 | `PROTOCOLS/overall_local_v1/` |
| 选样、评分回传与 UCB | `LLM/ucb_chemical_units_v1/` |
| 常规 GA 对接 | `ten_generation_native_ga.py` |

## 运行

从仓库根目录安装依赖。

```sh
python -m pip install -r requirements-matrix.txt networkx
```

用 `--inputs` 指向本地实验归档，或者使用包含下列相对路径的独立数据目录。

| 相对路径 | 内容 |
| --- | --- |
| `SOM/chemical_unit_som_v1/artifacts/assignments.npz` | `node` 和 `eligible` 数组 |
| `workflow_runtime_20261125/runtime/outputs/matrix_som_v2/matrix_inputs.npz` | 与 SOM 指派逐行对齐的 `x` 和 `mask` 数组 |
| `workflow_runtime_20261125/runtime/outputs/matrix_map_v1/data_split.csv` | 含 `lean_smiles` 的外源补入候选 |
| `LLM/overall_local_v1/artifacts/MEMORY_V2.json` | 局部化学单元证据 |
| `runs/frontier_small_001/memory_store/memory/global.json` | 含 `records` 的整体评分 Memory |

使用新的输出目录。省略 `--key` 和 `--resume` 时执行离线覆盖检查。

```sh
python dual_route/run_dual_ten_generation.py --inputs "<local-data-root>" --out runs/dual_route_coverage
```

用本地检查点完整离线回读，或配置 DeepSeek 凭据进行真实运行。

```sh
python dual_route/run_dual_ten_generation.py --inputs "<local-data-root>" --resume "<local-checkpoint>" --inspect-only --out runs/dual_route_replay
python dual_route/run_dual_ten_generation.py --inputs "<local-data-root>" --key "<local-key-file>" --out runs/dual_route_live
```

真实运行也可加 `--resume`，使用同一输入池、SOM 检查点和评分协议。运行目录含候选、分数、匹配证据和调用记录，应保存在本地。

原实验以 256 个候选开始，提交了 6 代。第 7 代取得 7/8 项有效评分并停在评分门槛。粗粒度 UCB 接收已验证评分，细粒度 GA 算子在混合候选批次中尚未单独记入收益。
