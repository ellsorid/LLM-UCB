# LOHC 矩阵迭代生成与贫氢态 GNN

最新的实验入口位于 [SOM–LLM–UCB 双链工作流](dual_route/README.md)，包含定向母体设计、种群扩展和评分回传。

新入口将双态矩阵、生成操作、QD、UCB、LLM评价和GPU种群连接起来。原有工作流保留在 `outputs/LOHC_From_Zero_Parallel_v10`，不替换主分支。

```bash
pip install -r requirements-matrix.txt
python -m unittest matrix_pipeline.test_v2 -v
python -m matrix_pipeline --bundled-only --device cpu --rounds 2 --output runs/offline_example
```

真实API运行：

```bash
python -m matrix_pipeline --bundled-only --rounds 3 --live --key-file '<已有凭据文件>' --output runs/live_example
```

默认关闭thinking、temperature=0。三轮通常最多8次请求：2次种子评价、3轮各一次选择和评价。不加 `--live` 时，种子与子代都使用明确标记的离线中性分数，不调用API。

## 主要文件

|内容|位置|
|---|---|
|执行方式、矩阵信息流与适用范围|[WORKFLOW](matrix_pipeline/WORKFLOW.md)|
|修复及验收说明|[验证报告](reports/MATRIX_V2_VERIFICATION.md)|
|机器可读验收结果|[validation.json](reports/matrix_loop_v2_validation.json)|
|单贫氢态12通道模型与五折权重|[模型卡](models/lean_matrix/MODEL_CARD.md)|
|权重SHA256及各折归一化|[checkpoint manifest](models/lean_matrix/checkpoint_manifest.json)|

仓库附带8个示例双态矩阵、冻结分格参数和现有文献memory，可在不连接原本地项目数据的情况下运行。完整9812对数据的审计及吞吐测试需要另行提供原项目数据。

v2新增贫富氢态成对侧链，使可释放H2数量能够变化；修正父代/操作顺序偏置、种子与子代评分口径，以及UCB缺少档案状态的问题。验收包括六项回归测试、三轮真实API和独立CPU冒烟测试。

这是有明确算子范围的研究版本。化学验证与旧SA仍在CPU边界，全部旧GA算子尚未迁移；独立稳定性验证和旧/新策略固定预算比较尚未完成。已有GNN权重原样保存，本次未重训，历史训练指标不代表新分子外推精度。
