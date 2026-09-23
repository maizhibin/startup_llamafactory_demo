# 创业精准孵化模型：LLaMA-Factory 可运行 Demo

这是一套**完全合成、无真实个人数据**的演示工程，用于把 A–F 六类业务数据加工成任务数据集，并生成可直接给 LLaMA-Factory 使用的 SFT/CPT 数据和训练配置。

## 1. 包含什么

### 原始示例数据（`data/raw/`）
- `A_founders.csv`：创业者身份与状态
- `B_employment_credit.csv`：就业与信用信息
- `C_projects.csv`：创业项目与主体
- `D_training_records.csv`：创业培训与能力
- `E_policies.csv`：创业政策与资源
- `F_finance_risk.csv`：金融与风险
- `F_investors.csv`：投资机构
- `F_market_signals.csv`：产业市场信号

### 加工后的核心任务数据集（`data/processed/`）
- `founder_profile.jsonl`
- `venture_stage_evaluation.jsonl`
- `skill_gap_training.jsonl`
- `policy_resource_matching.jsonl`
- `investment_matching.jsonl`
- `risk_opportunity.jsonl`

### LLaMA-Factory 数据
- `startup_sft_train.json`：8 项创业任务的 SFT 训练集
- `startup_sft_eval.json`：验证集
- `startup_cpt_demo.jsonl`：CPT 格式演示（仅用于格式/流程 smoke test，规模太小，不应当作为正式 CPT）
- `data/dataset_info.json`：LLaMA-Factory 数据注册文件

## 2. 先生成/验证数据

```bash
cd startup_llamafactory_demo
./run_generate.sh
```

脚本会固定随机种子重新生成全部样例，并校验：
- JSON / JSONL 是否可解析
- SFT 是否包含 instruction/input/output/system
- input/output 是否为合法 JSON 文本
- 训练集/验证集 8 个任务是否平衡
- YAML 是否可解析（系统有 PyYAML 时）

## 3. 安装 LLaMA-Factory

建议使用 LLaMA-Factory 官方当前版本。示例：

```bash
git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory
pip install -e ".[torch,metrics]"
```

然后将本 Demo 目录放到任意位置，或把 `data/`、`configs/` 拷贝到 LLaMA-Factory 工作目录。若从本 Demo 目录直接执行，请确保 `llamafactory-cli` 已在 PATH 中。

## 4. LoRA SFT（推荐先跑）

基座：`Qwen/Qwen3.5-4B`

```bash
cd startup_llamafactory_demo
CUDA_VISIBLE_DEVICES=0 ./run_train_lora.sh
```

等价命令：

```bash
llamafactory-cli train configs/qwen3_4b_lora_sft.yaml
```

### QLoRA（显存更紧张时）

```bash
CUDA_VISIBLE_DEVICES=0 ./run_train_qlora.sh
```

## 5. CPT 格式 smoke test

```bash
llamafactory-cli train configs/qwen3_4b_lora_cpt_demo.yaml
```

注意：本包只有几十条合成 CPT 文本，只用于验证流程。正式项目应从 2.07TB 数据资产中筛选高质量领域文本，再按 token 规模重新设计数据配比。

## 6. 推理与合并 LoRA

```bash
./run_chat.sh
./run_merge.sh
```

## 7. 8 个 SFT 任务

1. 创业人员画像
2. 创业能力提升
3. 创业机会识别
4. 创业路径规划
5. 创业计划书生成
6. 创业项目评估
7. 创业资源匹配
8. 创业风险评估

每条 SFT 样本都只学习**可展示的依据、结构化结论与建议**，不依赖隐藏推理过程。

## 8. 从 Demo 升级到真实项目时要改什么

1. 用真实 A–F 数据接入脚本替换 `generate_demo_dataset.py` 中的合成数据生成部分。
2. 所有事实数据增加 `source_id / snapshot_time / valid_from / valid_to / quality_score / label_source / data_version`。
3. 训练/验证/测试按**主体 + 时间**切分，避免同一人/企业跨集合泄漏。
4. 项目评估、风险、推荐等标签优先级：`真实业务结果 > 专家标注 > 规则生成 > LLM 合成`。
5. 政策资格判断保留规则引擎/RAG；不要强迫 LLM 记忆时效性政策。
6. 信用、融资等高影响场景只能做辅助分析；不要把敏感/无关个人属性作为自动决策特征。

## 9. 目录

```text
startup_llamafactory_demo/
├── data/
│   ├── dataset_info.json
│   ├── raw/
│   └── processed/
├── configs/
├── scripts/
├── run_generate.sh
├── run_train_lora.sh
├── run_train_qlora.sh
├── run_chat.sh
├── run_merge.sh
└── README.md
```
