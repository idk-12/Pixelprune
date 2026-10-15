# MMTok × Qwen3.5 视觉 Token 压缩实验

## 实验目标

在 Qwen3.6-27B（`Qwen3_5ForConditionalGeneration`）上集成 MMTok（training-free 视觉 token 压缩），
量化不同压缩率下的**推理加速比**与**精度损失**，在 MMBench 上完成 baseline vs MMTok 对比。

---

## 方法介绍：MMTok

**论文/代码**：https://github.com/Ironieser/MMTok

MMTok 是一种 training-free 视觉 token 选取方法，作用于 ViT 输出之后、LLM 输入之前：

- **Text-Vision Coverage**：选取能最好覆盖问题关键词语义的视觉 token
- **Vision-Vision Diversity**：选取能代表整体图像分布的 token（避免冗余）
- 贪心最大化两者联合覆盖，保留 `retain_ratio * N` 个视觉 token
- 完全 training-free，不修改模型权重

原生支持 Qwen2.5-VL；本实验适配至 Qwen3.5（`Qwen3_5ForConditionalGeneration`）。

---

## 模型架构说明

| 项目 | 值 |
|---|---|
| 模型路径 | `/mnt/nfs/data/pretrained_models/Qwen3.6-27B` |
| HuggingFace 模型类 | `Qwen3_5ForConditionalGeneration` |
| config model_type | `qwen3_5` |
| 视觉编码器 | `Qwen3_5VisionModel`（复用 Qwen3-VL 视觉组件，depth=27, hidden=1152） |
| LLM backbone | 混合注意力（linear_attention + full_attention + gated_delta_net） |
| 位置编码 | M-RoPE（需要 `mm_token_type_ids`） |
| image_token_id | 248056 |
| video_token_id | 248057 |
| VLMEvalKit wrapper | `Qwen3_5Chat`（`vlmeval/vlm/qwen3_5/model.py`） |

---

## 实验配置

### 模型变体

| 模型名 | 推理后端 | retain_ratio | 说明 |
|---|---|---|---|
| `Qwen3.6-27B-base` | transformers | — | baseline（无 token 压缩） |
| `Qwen3.6-27B-MMTok-r50` | transformers | 0.5 | 保留 50% 视觉 token |
| `Qwen3.6-27B-MMTok-r30` | transformers | 0.3 | 保留 30% 视觉 token |
| `Qwen3.6-27B-MMTok-r20` | transformers | 0.2 | 保留 20% 视觉 token |

### Benchmark

- **MMBench_DEV_EN**（约 4377 题，多项选择，综合视觉理解）

---

## 实现记录

### 环境

- conda 环境：`vllm`
- transformers 版本：`4.57.1`（随模型 config 携带）
- MMTok 安装路径：`/data/lijie/MMTok`（`pip install -e .`）

---

### 文件变更清单

#### 新增文件

| 文件 | 说明 |
|---|---|
| `vlmeval/vlm/qwen3_5/model.py` | Qwen3.5 专用 wrapper：`Qwen3_5Chat`、`Qwen3_5ChatMMTok` |
| `vlmeval/vlm/qwen3_5/__init__.py` | 导出上述两个类 |
| `/data/lijie/MMTok/mmtok/qwen/qwen3_5_VLmodel_mmtok.py` | Qwen3.5 的 model-level patch 类（`Qwen3_5_MMTok`） |
| `/data/lijie/MMTok/mmtok/qwen/qwen3_5_vl_mmtok.py` | 注入函数 `mmtok_qwen3_5_vl()` |

#### 修改文件

| 文件 | 改动 |
|---|---|
| `/data/lijie/MMTok/mmtok/qwen/__init__.py` | 导出 `mmtok_qwen3_5_vl` |
| `vlmeval/vlm/__init__.py` | 导入 `Qwen3_5Chat`、`Qwen3_5ChatMMTok` |
| `vlmeval/config.py` | 注册 `Qwen3.6-27B-base` 及三个 MMTok 变体，改用 `Qwen3_5Chat`/`Qwen3_5ChatMMTok` |
| `vlmeval/inference.py` | 添加 `time` / `json` import，推理循环加计时，结果写入 `*_timing.json` |

---

### 为什么需要专用 wrapper（Qwen3_5Chat 而非 Qwen3VLChat）

`Qwen3VLChat` 设计用于 `qwen3_vl` 架构（`Qwen3VLForConditionalGeneration`），与 Qwen3.5 的
`Qwen3_5ForConditionalGeneration`（`qwen3_5` 架构）存在以下关键差异：

| 差异点 | Qwen3VLChat（旧） | Qwen3_5Chat（新） |
|---|---|---|
| 输入处理 | `qwen_vl_utils.process_vision_info` + `processor(do_resize=False)` | `processor.apply_chat_template(tokenize=True, return_dict=True)` |
| `mm_token_type_ids` 产生 | 依赖 qwen_vl_utils 的两步流程，有兼容风险 | processor 内置，自动产生，官方推荐做法 |
| 图片 URL 格式 | `file://` 前缀（`ensure_image_url`） | 原始文件路径（PIL 直接 `open`，`file://` 会报错） |
| LLM backbone | 标准多头注意力 | 混合注意力（linear + full + gated_delta_net） |

---

### MMTok 各文件职责与调用关系

#### 文件分层

```
【框架层】vlmeval/vlm/qwen3_5/model.py
  Qwen3_5Chat        ─ 加载模型/processor，格式化 benchmark 输入，调 generate
  Qwen3_5ChatMMTok   ─ 继承 Qwen3_5Chat，__init__ 时调用 mmtok_qwen3_5_vl() 完成注入

【注入层】/data/lijie/MMTok/mmtok/qwen/qwen3_5_vl_mmtok.py
  mmtok_qwen3_5_vl() ─ 把 MMTok 逻辑挂载到已加载的模型（monkey-patch，只执行一次）

【算法层】/data/lijie/MMTok/mmtok/qwen/qwen3_5_VLmodel_mmtok.py
  Qwen3_5_MMTok      ─ 定义替换后的 forward / get_image_features（每次推理时执行）

【核心算法】/data/lijie/MMTok/mmtok/core/
  mmtok_core.py      ─ 文本编码、调用 SemanticTokenSelector
  semantic_selector.py ─ 构造 P/Q 矩阵，贪心选 token（真正的选择算法）
```

#### 初始化时的调用链（一次性）

```
Qwen3_5ChatMMTok.__init__()
  └─ mmtok_qwen3_5_vl(self.model, tokenizer, processor, retain_ratio)
       ├─ 校验模型类型为 Qwen3_5ForConditionalGeneration
       ├─ 创建 MMTokCore（含 SemanticTokenSelector + VQATextProcessor）
       ├─ inner = model.model  (Qwen3_5Model)
       ├─ inner.forward          ← 替换为 Qwen3_5_MMTok.forward
       ├─ inner.get_image_features ← 替换为 Qwen3_5_MMTok.get_image_features
       └─ processor.apply_chat_template ← 打补丁，推理时拦截问题文本
```

#### 每次推理时的调用链

```
Qwen3_5Chat.generate_inner(message)
  │
  ├─ processor.apply_chat_template(tokenize=True, return_dict=True)
  │     └─ [MMTok 补丁] 从 messages 提取问题文本
  │                     → model.model._question_for_vision = question
  │
  └─ model.generate(**inputs)
       └─ model.model.forward()   ← 已替换为 Qwen3_5_MMTok.forward
             │
             ├─ get_image_features(pixel_values, image_grid_thw)
             │     └─ visual(pixel_values)  →  BaseModelOutputWithPooling
             │          image_embeds  = pooler_output      [N, D]  post-merger → 插入 LLM
             │          img_features  = last_hidden_state 均值 [N, d]  pre-merger → 算相似度
             │
             ├─ inputs_embeds.masked_scatter(image_mask, image_embeds)
             │     # 把 N 个图像 token 填入完整序列
             │
             ├─ compute_3d_position_ids(...)  # 在完整序列上算 M-RoPE
             │
             ├─ ← MMTok token 选择在此发生 ←
             │   _mmtok_core.apply_selection_preprocess_qwen(
             │       image_embeds, img_features, question,
             │       target_vision_tokens = retain_ratio * N)
             │     └─ MMTokCore → SemanticTokenSelector.mm_coverage_selection()
             │           P = text_embed @ image_embed.T   # 文本-视觉 coverage
             │           Q = image_embed @ image_embed.T  # 视觉-视觉 diversity
             │           Combined = [P ; Q*alpha]
             │           greedy_merged_jit_kernel(Combined, k)  # 贪心选 k 个 token
             │
             ├─ 删除未选中的图像 token（inputs_embeds / attention_mask / position_ids 同步 slice）
             │     # 序列长度：(n_text + N) → (n_text + k)
             │
             └─ language_model(inputs_embeds=pruned_embeds, ...)  # 用压缩序列跑 LLM
```

#### Token 数量变化全流程

```
原始图片
  → patch_embed：H/16 × W/16 × 4 个 patch        例：40×30×4 = 4800 patches
  → 27层 VisionBlock
  → Qwen3_5VisionPatchMerger（spatial_merge_size=2，固定 4→1）
                                                   → 600 个 merged token（post-merger）
  → inputs_embeds 中填入 600 个图像 token
  → MMTok 选择 retain_ratio × 600 个
    r=0.5 → 300   r=0.3 → 180   r=0.2 → 120
  → LLM 处理压缩后的序列
```

---

### 关键技术决策

#### 1. 无需 patch `visual.forward`

`Qwen3_5VisionModel.forward` 原生返回 `BaseModelOutputWithPooling`：
- `pooler_output` = `merger(hidden_states)` → shape `[N_merged, D]`（post-merger，即 LLM 接收的视觉 token）
- `last_hidden_state` = `hidden_states` → shape `[4·N_merged, d]`（pre-merger patch 特征）

MMTok 需要两者：`pooler_output` 作为插入 LLM 的 embedding，`last_hidden_state` 平均后用于计算 text-vision coverage。Qwen2.5-VL 则需要额外 patch visual forward 才能拿到 pre-merger 特征，Qwen3.5 天然支持。

#### 2. `get_image_features` 返回 flat tensor

标准 `Qwen3_5Model.get_image_features()` 对结果按图片 split 成 tuple，调用方再 cat。
我们的 patch 直接返回 `(image_embeds, img_features)` flat tensor，避免 split-cat 的冗余：

```python
def get_image_features(self, pixel_values, image_grid_thw=None, **kwargs):
    outputs = self.visual(pixel_values, grid_thw=image_grid_thw, return_dict=True)
    image_embeds = outputs.pooler_output          # [N_merged, D]
    pre_merger = outputs.last_hidden_state        # [4·N_merged, d]
    smu = self.visual.spatial_merge_unit          # =4 (spatial_merge_size=2)
    img_features = pre_merger.reshape(-1, smu, pre_merger.shape[-1]).mean(dim=1)
    return image_embeds, img_features
```

**踩坑**：初版误写 `torch.cat(list(outputs.pooler_output), dim=0)`，`list()` 作用在 2D tensor 上会按行拆分，cat 后变成展平的 1D tensor（`[327680]` vs 正确的 `[64, D]`），导致 token 数量校验失败。直接用 `outputs.pooler_output` 即可。

#### 3. M-RoPE 处理：先算再 slice

Qwen3.5 使用 M-RoPE（需 `mm_token_type_ids`），`compute_3d_position_ids()` 必须在完整序列上计算。

策略：先用完整的 `mm_token_type_ids` 计算 `position_ids`（3D，shape `[3, batch, seq_len]`），再在 MMTok 选 token 后对 `inputs_embeds`、`attention_mask`、`position_ids` 同步 slice。被保留的视觉 token 保持其原始 3D 位置坐标，文本 token 的 1D 位置不变。

#### 4. 与 Qwen2.5-VL 的接口差异

| 差异点 | Qwen2.5-VL | Qwen3.5 |
|---|---|---|
| 模型类 | `Qwen2_5_VLForCausalLM` | `Qwen3_5ForConditionalGeneration` |
| 内层模型类 | `Qwen2_5_VLModel` | `Qwen3_5Model` |
| 视觉模型类 | `Qwen2_5_VisionTransformerPretrainedModel` | `Qwen3_5VisionModel` |
| visual.forward 返回 | `hidden_states`（仅 merged） | `BaseModelOutputWithPooling`（含 pre+post merger） |
| 位置编码 | `get_rope_index()` 直接调用 | `compute_3d_position_ids()` 封装（需 `mm_token_type_ids`） |
| 输出类型 | `Qwen2_5_VLModelOutputWithPast` | `Qwen3_5ModelOutputWithPast` |
| image_token_id | 151655 | 248056 |
| visual.get_window_index | ✓ | ✗（无窗口注意力） |
| visual.fullatt_block_indexes | ✓ | ✗ |

---

### VLMEvalKit 集成

`Qwen3_5ChatMMTok` 继承 `Qwen3_5Chat`，加载模型后调用 `mmtok_qwen3_5_vl()` 完成 patch：

```python
class Qwen3_5ChatMMTok(Qwen3_5Chat):
    def __init__(self, model_path, retain_ratio=0.5, **kwargs):
        super().__init__(model_path, **kwargs)
        from mmtok.qwen.qwen3_5_vl_mmtok import mmtok_qwen3_5_vl
        mmtok_qwen3_5_vl(
            self.model, self.processor.tokenizer,
            self.processor, retain_ratio=retain_ratio,
        )
        self.retain_ratio = retain_ratio
```

`generate_inner` 调用 `processor.apply_chat_template(tokenize=True, return_dict=True)`，
该调用已被 MMTok 补丁拦截，自动提取问题文本并存入模型供 token 选择使用。

### 计时方案

在 `vlmeval/inference.py` 的 `infer_data()` 循环中，用 `time.perf_counter()` 包裹每次 `model.generate()` 调用，推理结束后写入：

```
outputs/{model}/{timestamp}/{rank}_{dataset}_timing.json
```

字段：`count`、`total_seconds`、`mean_seconds`、`per_sample`（各样本耗时列表）。

---

## 运行方式

```bash
# 依次运行（每个约需数小时，27B 模型单 GPU 不能并行）
conda run -n vllm python run.py --data MMBench_DEV_EN --model Qwen3.6-27B-base --mode infer
conda run -n vllm python run.py --data MMBench_DEV_EN --model Qwen3.6-27B-MMTok-r50 --mode infer
conda run -n vllm python run.py --data MMBench_DEV_EN --model Qwen3.6-27B-MMTok-r30 --mode infer
conda run -n vllm python run.py --data MMBench_DEV_EN --model Qwen3.6-27B-MMTok-r20 --mode infer

# infer 完成后跑 judge 评测（去掉 --mode infer）
conda run -n vllm python run.py --data MMBench_DEV_EN --model Qwen3.6-27B-base
# ...（其他变体同理）
```

---

## 测评流程（输入输出）

### 总览

```
compare_mmtok.sh
  └─ run.py × N_models（串行）
        ├─ infer_data_job()   →  predictions CSV  +  _timing.json
        └─ dataset.evaluate() →  accuracy CSV/xlsx
  └─ 内嵌 Python 汇总 → summary_<timestamp>.txt
```

---

### Step 1 — `compare_mmtok.sh`（调度层）

| | 内容 |
|---|---|
| **输入** | 命令行参数 `N`（样本数，默认 0=全量）；脚本内 `MODELS` 列表；`DATA=TextVQA_VAL` |
| **操作** | 对每个模型顺序调用 `python run.py` |
| **输出** | 触发下游各步骤；最终生成 `./outputs/textvqa/summary_YYYYMMDD_HHMMSS.txt` |

---

### Step 2 — `run.py`（流程编排，单模型单数据集）

| | 内容 |
|---|---|
| **输入** | `--data TextVQA_VAL  --model $MODEL  --max-samples $N  --judge exact_matching  --work-dir ./outputs/textvqa` |
| **操作** | 构建模型实例、加载 dataset；设置输出目录 `./outputs/textvqa/{model_name}/{eval_id}/`；调用 `infer_data_job()`，再调用 `dataset.evaluate()` |
| **输出** | 无直接产物；由下游两步产生文件 |

---

### Step 3 — `infer_data_job()` / `infer_data()` （`vlmeval/inference.py`，推理层）

| | 内容 |
|---|---|
| **输入** | dataset TSV（从 `~/LMUData/` 自动下载）；已加载的 model 实例 |
| **核心循环** | 对每条样本调用 `model.generate(message, dataset)`，用 `time.perf_counter()` 前后计时 |
| **中间产物** | `{work_dir}/{rank}{world_size}_{dataset}.pkl` — 各 rank 的 `{index: response}` dict（多 GPU 时每 rank 一份） |
| **Rank-0 合并** | 读取所有 rank 的 pkl，将 `prediction` 列追加到原始 TSV DataFrame，去掉 `image` 列（base64），写成完整预测文件 |
| **推理产物** | `{work_dir}/{model_name}_{dataset_name}.csv`（或 `.tsv`/`.xlsx`，含 `prediction` 列） |
| **计时产物** | `{work_dir}/{rank}{world_size}_{dataset}_timing.json`，字段：`count`、`total_seconds`、`mean_seconds`、`per_sample` |

---

### Step 4 — `dataset.evaluate(result_file)`（评测层）

以 TextVQA 为例：

| | 内容 |
|---|---|
| **输入** | Step 3 产出的 CSV，含原始题目字段 + `prediction` 列 |
| **操作** | `--judge exact_matching` 时直接字符串比对答案；计算 Overall 准确率 |
| **产物** | `{work_dir}/…/*TextVQA*.csv`（或 `.xlsx`），含 `Overall` 列，末行为汇总分数 |
| **返回值** | `dict` 或 `pd.DataFrame`，由 `run.py` 打印到日志 |

---

### Step 5 — 内嵌 Python 汇总（`compare_mmtok.sh` 末段）

| | 内容 |
|---|---|
| **输入** | 所有模型目录下最新的 `*_timing.json`（计时）和 `*TextVQA*.csv/xlsx`（精度） |
| **操作** | 以第一个有计时数据的模型为基线，计算 `speedup = base_mean_s / model_mean_s` |
| **产物** | 终端打印对比表（Model / Acc / Mean(s) / Total(s) / Speedup）；同时写入 `./outputs/textvqa/summary_<timestamp>.txt` |

---

### 文件路径速查

```
./outputs/textvqa/
├── {MODEL}/
│   └── {eval_id}/                         ← eval_id = 时间戳
│       ├── {rank}{ws}_{dataset}.pkl        ← 各 rank 推理中间结果（合并后自动删除）
│       ├── {MODEL}_{DATASET}.csv           ← 含 prediction 列的完整预测文件
│       ├── {rank}{ws}_{dataset}_timing.json← 计时统计
│       └── *TextVQA*.csv / .xlsx           ← 精度评测结果（含 Overall）
└── summary_YYYYMMDD_HHMMSS.txt            ← 多模型对比汇总表
```

---

## 结果（待填写）

| 模型 | MMBench Acc (%) | 均值时间/样本 (s) | 加速比 |
|---|---|---|---|
| Qwen3.6-27B-base | — | — | 1.0× |
| MMTok r=0.5 | — | — | — |
| MMTok r=0.3 | — | — | — |
| MMTok r=0.2 | — | — | — |

计时数据从 `outputs/{model}/{timestamp}/*_timing.json` 读取，精度从 MMBench evaluate 输出读取。
