# Gemma 4 26B-A4B-it 在 Ascend 910B3 NPU 上的 vLLM 推理服务部署计划

> 创建日期：2026-04-11
> 目标：在单卡 Ascend 910B3 (64GB HBM) 上通过 vLLM-Ascend 部署 Gemma 4 26B-A4B-it 推理服务

---

## 一、当前环境

| 组件 | 当前版本 | 目标版本 |
|------|---------|---------|
| NPU | Ascend 910B3 (单卡, 64GB HBM) | 不变 |
| CANN | 8.5.0 (`/home/developer/Ascend/cann-8.5.0/`) | 8.5.0（v0.13.0 兼容） |
| 驱动 | 25.5.0 | 不变 |
| Python | 3.11.4 | 不变 |
| PyTorch | 2.7.1+cpu | **2.8.0** |
| torch_npu | 2.7.1.post2.dev20251226 | **2.8.0.post2** |
| vLLM | 未安装 | **0.13.0** |
| vLLM-Ascend | 未安装 | **0.13.0** |
| transformers | 5.5.3（venv 中） | >=4.57.4, <5（v0.13.0 要求） |
| triton-ascend | 未安装 | **3.2.0** |

### 已验证的能力

- Gemma 4 模型权重已下载至 `/mnt/workspace/models/google/gemma-4-26B-A4B-it/`（约 49GB）
- 使用 transformers 5.5.3 + torch_npu 已成功在 NPU 上加载并推理 Gemma 4（文本生成正常）
- venv 环境已创建于 `/mnt/workspace/venv`（带 --system-site-packages）

---

## 二、核心技术挑战

### 2.1 CANN 算子限制（最关键）

CANN 的融合注意力算子仅支持 `head_dim ∈ {64, 128, 192}`，而 Gemma 4 使用：
- sliding attention 层（25/30 层）：`head_dim=256`
- full attention 层（5/30 层）：`global_head_dim=512`

**解决方案**：PyTorch SDPA（scaled_dot_product_attention）回退路径，放弃 CANN 融合注意力，使用 eager 模式。

**性能影响**：推理速度会明显低于使用 CANN 融合注意力的模型，但功能可用。

### 2.2 MoE 架构（128 专家，Top-8 路由）

Gemma 4 26B-A4B-it 是 MoE 模型，每层有 128 个专家，每次激活 8 个。

**挑战**：
- vLLM-Ascend 的 MoE 算子需要适配 `npu_moe_init_routing_v2`
- 专家权重以 3D 参数存储 `[128, ...]`
- 路由器有 per-expert scale 参数
- 自定义路由函数签名兼容性（`global_num_experts` 参数）

### 2.3 异构注意力层

不同层有不同的 head_dim 和 num_kv_heads：
- sliding 层：head_dim=256, num_kv_heads=8
- full attention 层：head_dim=512, num_kv_heads=2

**挑战**：vLLM 的 PagedAttention 假设所有层的 KV cache 形状一致。

### 2.4 其他非标准特性

| 特性 | 说明 |
|------|------|
| K=V 共享 | Layer 29 没有 v_proj，key 输出直接作为 value |
| Proportional RoPE | full attention 层仅 25% 维度应用 RoPE |
| layer_scalar | 每层有可学习的标量乘数 |
| 自定义视觉编码器 | 非 SigLIP，使用多维 RoPE + 可学习位置嵌入 |
| 双向注意力 | 视觉 token 使用双向注意力 |

---

## 三、vLLM-Ascend PR #8081 分析

### 3.1 PR 概要

| 项目 | 详情 |
|------|------|
| PR 编号 | #8081 |
| 标题 | [Core][Test][Doc] Add Gemma4 support on Ascend |
| 作者 | EheinWang |
| 状态 | Open（未合并） |
| 目标分支 | vllm-project:main（对应 vLLM v0.19.0） |
| 文件变更 | 24 个文件 |
| 验证模型 | gemma-4-E2B-it, E4B-it, 31B-it, **26B-A4B-it** |
| 测试硬件 | Ascend 910 NPU, torch 2.10.0, torch_npu 2.10.0rc2 |

### 3.2 核心改动文件（10 个）

| 文件 | 改动内容 |
|------|---------|
| `vllm_ascend/attention/attention_v1.py` | SDPA 回退（head_dim 256/512）、共享 KV 处理、禁用 cudagraph |
| `vllm_ascend/patch/worker/patch_gemma4_transformers.py` | **新文件** - Gemma4 transformers monkey patch（KV 共享、MM token、注意力实例、embed 缩放） |
| `vllm_ascend/patch/worker/__init__.py` | 注册 Gemma4 patch |
| `vllm_ascend/ops/fused_moe/experts_selector.py` | 自定义路由函数签名兼容（global_num_experts） |
| `vllm_ascend/ops/fused_moe/fused_moe.py` | 保留原始 gate/shared_experts 模块 |
| `vllm_ascend/ops/layernorm.py` | 强制 Gemma4 使用原生 RMSNorm |
| `vllm_ascend/device/device_op.py` | 优先使用 npu_moe_init_routing_v2 |
| `vllm_ascend/platform.py` | 强制 eager 模式 + 禁用 cudagraph |
| `vllm_ascend/worker/model_runner_v1.py` | 多模态输入选择逻辑 |
| `vllm_ascend/worker/npu_input_batch.py` | logprob token id 追踪 |

### 3.3 已知 Bug（需修复）

1. **注意力掩码因果性反转**（严重）：布尔掩码构建为"允许"语义（True=保留），但 `scaled_dot_product_attention` 将 True 视为"遮蔽"（排除），导致行为反转
2. **`inspect.signature()` 性能问题**：在每次 forward 中对每个 MoE 层调用，开销大，应缓存结果
3. **`inspect.signature()` 异常风险**：对内置函数/C 扩展可能抛出 ValueError/TypeError，需 try/except
4. **forward 路径中的 host-to-device 同步**：从 Python list 创建 tensor 导致同步瓶颈

### 3.4 版本差距

PR #8081 基于 vLLM v0.19.0（main 分支），而我们计划使用 v0.13.0。主要 API 差异：
- attention 接口可能不同（v1 vs v2 backend）
- MoE 算子接口可能变化
- worker/model_runner 结构可能重构
- patch 机制可能不同

---

## 四、实施计划

### 阶段 1：环境升级与基础验证（预计 1-2 天）

#### 1.1 升级 PyTorch 和 torch_npu

```bash
# 卸载旧版本
pip uninstall torch torch_npu -y

# 安装 CANN 8.5.0 对应的 PyTorch 2.8.0 + torch_npu 2.8.0.post2
# 参考：https://github.com/Ascend/pytorch
pip install torch==2.8.0
pip install torch_npu==2.8.0.post2

# 验证
python3 -c "import torch; import torch_npu; print(torch.npu.is_available())"
```

**注意**：需要确认 CANN 8.5.0 下 PyTorch 2.8.0 的 wheel 包是否可用。如果不可用，可能需要从源码编译或使用华为提供的预编译包。

#### 1.2 安装 vLLM 和 vLLM-Ascend

```bash
# 方案 A：pip 安装（推荐）
pip install vllm==0.13.0
pip install vllm-ascend==0.13.0
pip install triton-ascend==3.2.0

# 方案 B：从源码安装（如果 pip 不可用）
# 设置环境变量
export ASCEND_HOME_PATH=/home/developer/Ascend/cann-8.5.0
export SOC_VERSION=ascend910b1  # 或 ascend910b3

# 克隆并编译
git clone -b releases/v0.13.0 https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
pip install -e .
```

#### 1.3 验证基础推理服务

```bash
# 用已支持的模型（如 Qwen2.5-7B）验证 vLLM-Ascend 能正常工作
python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --port 8000
```

**验证项**：
- [ ] PyTorch 2.8.0 + torch_npu 2.8.0.post2 正常工作
- [ ] vLLM-Ascend v0.13.0 安装成功
- [ ] NPU 被正确识别
- [ ] 已支持模型能正常推理
- [ ] OpenAI 兼容 API 服务正常响应

**风险点**：
- Ascend 910B3 可能不在 vLLM-Ascend 的设备白名单中，需要检查是否需要 patch 设备识别逻辑
- CANN 8.5.0 + PyTorch 2.8.0 的组合可能需要华为特定的 wheel 包

---

### 阶段 2：移植 Gemma 4 支持（预计 3-5 天）

#### 2.1 准备工作

```bash
# 克隆 vLLM-Ascend v0.13.0 源码（用于修改）
git clone -b releases/v0.13.0 https://github.com/vllm-project/vllm-ascend.git \
    /mnt/workspace/vllm-ascend-gemma4

# 获取 PR #8081 的 patch
cd /mnt/workspace/vllm-ascend-gemma4
git fetch origin pull/8081/head:pr-8081
```

#### 2.2 逐文件适配（按优先级排序）

**文件 1：`vllm_ascend/platform.py`** — 强制 eager 模式

改动最小，独立性强。检测 Gemma4 模型时强制 `enforce_eager=True` 和禁用 cudagraph。

适配要点：
- 检查 v0.13.0 的 platform.py 结构是否与 PR 中一致
- `_fix_incompatible_config()` 方法是否存在
- CUDAGraphMode 枚举名称是否一致

**文件 2：`vllm_ascend/ops/layernorm.py`** — 原生 RMSNorm

改动小。添加 `_is_gemma4_text_model()` 检查，强制走原生路径。

适配要点：
- AscendRMSNorm 类名和方法签名是否一致
- `forward_oot` 方法是否存在

**文件 3：`vllm_ascend/device/device_op.py`** — MoE 路由 v2

改动小。添加 Gemma4 检测，优先使用 `npu_moe_init_routing_v2`。

适配要点：
- `npu_moe_init_routing` 方法签名
- v2 API 在 torch_npu 2.8.0.post2 中是否可用

**文件 4：`vllm_ascend/ops/fused_moe/experts_selector.py`** — 路由函数签名

改动中等。添加 `_call_custom_routing_function()` 处理 `global_num_experts` 参数。

适配要点：
- experts_selector.py 在 v0.13.0 中的结构
- 需要修复 `inspect.signature()` 的性能和异常问题

**文件 5：`vllm_ascend/ops/fused_moe/fused_moe.py`** — 保留原始模块

改动中等。缓存 gate/shared_experts 模块引用。

适配要点：
- AscendFusedMoE 类的 __init__ 签名
- AscendMoERunner 的初始化参数

**文件 6：`vllm_ascend/attention/attention_v1.py`** — SDPA 回退（最复杂）

改动最大。添加完整的 SDPA 回退路径，包括：
- `_is_gemma4_model()` 检测
- `_forward_gemma4_torch_attention()` 核心回退方法
- `_gather_paged_kv_to_dense()` 从分页 KV cache 收集到密集格式
- `_expand_kv_for_dense_attention()` GQA KV 头扩展
- 共享 KV cache 处理
- 注意力掩码构建（**必须修复因果性反转 bug**）

适配要点：
- v0.13.0 的 attention backend 是 v1 还是 v2
- AscendAttentionMetadataBuilder 的接口
- Paged KV cache 的数据结构
- **关键修复**：掩码语义必须正确（True = masked out for SDPA）

**文件 7：`vllm_ascend/patch/worker/patch_gemma4_transformers.py`** — 新文件

这是最大的新增文件，包含所有 transformers monkey patch：
- `_patch_config()` — 设置 attn_implementation
- `_gemma4_embed_forward()` — embedding 缩放
- `_forward_gemma4_text_attention_for_vllm_shared_kv()` — 共享 KV attention forward
- `_gemma4_text_attention_forward()` — attention forward 路由
- `_gemma4_text_model_forward()` — 强制 use_cache=True
- `_gemma4_model_forward()` — 多模态 forward（最复杂）
- `_init()` — 初始化 patch
- `_causal_init()` — logits softcapping
- `_multimodal_forward()` — MM token type ids 传递
- `_create_attention_instances()` — 异构注意力实例构建

适配要点：
- v0.13.0 的 patch 机制（`vllm_ascend/patch/worker/` 目录结构）
- transformers 版本差异（v0.13.0 要求 <5，但 Gemma4 需要 5.5+）
- **这是一个重大冲突**：v0.13.0 的 transformers 依赖是 `>=4.57.4, <5`，但 Gemma4 需要 transformers >=5.5.0

**文件 8：`vllm_ascend/patch/worker/__init__.py`** — 注册 patch

改动极小，添加一行 import。

**文件 9：`vllm_ascend/worker/model_runner_v1.py`** — 多模态输入

改动中等。集中处理 MM raw tokens / MM embeds / prompt embeds 的选择逻辑。

**文件 10：`vllm_ascend/worker/npu_input_batch.py`** — logprob 追踪

改动小。

#### 2.3 transformers 版本冲突处理

**问题**：vLLM v0.13.0 要求 `transformers>=4.57.4,<5`，但 Gemma4 需要 `transformers>=5.5.0`。

**解决方案**：
1. **方案 A（推荐）**：升级 transformers 到 5.5.x，忽略 vLLM 的版本约束，测试是否兼容。vLLM 对 transformers 的版本上限通常是保守估计，实际可能可以工作。
2. **方案 B**：fork vLLM v0.13.0，修改 pyproject.toml 放开 transformers 版本约束。
3. **方案 C**：使用 vLLM 的 `--trust-remote-code` 选项，让 transformers 从模型目录加载自定义代码。

#### 2.4 Bug 修复清单

在移植过程中，必须修复 PR #8081 中的已知 bug：

1. **注意力掩码因果性反转**：
   ```python
   # 原始（错误）：构建为"允许"掩码
   attn_mask = key_positions <= query_positions  # True = 允许

   # 修复：取反，构建为"遮蔽"掩码（SDPA 语义）
   attn_mask = ~(key_positions <= query_positions)  # True = 遮蔽
   ```

2. **`inspect.signature()` 缓存**：
   ```python
   # 修复：缓存签名检查结果
   if not hasattr(custom_routing_function, '_supports_global_experts'):
       sig = inspect.signature(custom_routing_function)
       supports = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
       supports = supports or 'global_num_experts' in sig.parameters
       custom_routing_function._supports_global_experts = supports
   ```

3. **`inspect.signature()` 异常处理**：
   ```python
   try:
       sig = inspect.signature(custom_routing_function)
   except (ValueError, TypeError):
       custom_routing_function._supports_global_experts = False
   ```

4. **host-to-device 同步优化**：
   ```python
   # 原始（慢）：从 Python list 创建
   seq_lens_tensor = torch.tensor(seq_lens, device=device)

   # 修复：直接使用 attn_metadata 中已有的 tensor
   seq_lens_tensor = attn_metadata.seq_lens_tensor
   ```

---

### 阶段 3：验证与调试（预计 2-3 天）

#### 3.1 文本推理验证

```bash
# 启动 vLLM 服务
python3 -m vllm.entrypoints.openai.api_server \
    --model /mnt/workspace/models/google/gemma-4-26B-A4B-it \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9 \
    --enforce-eager \
    --trust-remote-code \
    --port 8000

# 测试
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/mnt/workspace/models/google/gemma-4-26B-A4B-it",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 128
    }'
```

#### 3.2 验证清单

- [ ] 模型能成功加载到 NPU
- [ ] 文本生成结果正确（与 transformers 基线对比）
- [ ] 多轮对话正常
- [ ] OpenAI 兼容 API 正常
- [ ] 显存使用合理（预期 ~48GB 模型 + KV cache）
- [ ] 推理速度可接受（eager 模式下预期 5-10 tok/s）
- [ ] 多模态（图片）推理正常
- [ ] 长序列（>1024 tokens）正常（sliding window 切换）
- [ ] 并发请求处理正常

#### 3.3 性能基准

```bash
# 使用 vLLM 自带的 benchmark 工具
python3 benchmarks/benchmark_serving.py \
    --model /mnt/workspace/models/google/gemma-4-26B-A4B-it \
    --backend openai \
    --base-url http://localhost:8000 \
    --dataset-name random \
    --random-input-len 128 \
    --random-output-len 128 \
    --num-prompts 50
```

---

### 阶段 4（备选）：transformers + FastAPI 轻量服务

如果阶段 1-3 遇到不可逾越的障碍（如 PyTorch 升级失败、vLLM-Ascend 不兼容 910B3），则退回此方案。

#### 4.1 架构

```
FastAPI 服务
  └── 推理引擎（transformers + torch_npu）
       ├── 模型加载（bfloat16, device_map="auto"）
       ├── 文本生成（model.generate）
       └── 多模态处理（processor）
```

#### 4.2 实现要点

```python
# 核心推理逻辑（已验证可用）
from transformers import AutoModelForCausalLM, AutoProcessor
import torch, torch_npu

model = AutoModelForCausalLM.from_pretrained(
    model_path, dtype=torch.bfloat16, device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_path)

# 文本推理
messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
inputs = processor.apply_chat_template(messages, tokenize=True,
                                        return_dict=True, add_generation_prompt=True)
inputs = {k: v.to("npu:0") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
outputs = model.generate(**inputs, max_new_tokens=256)
```

#### 4.3 限制

- 无 continuous batching（一次只能处理一个请求）
- 无 PagedAttention（KV cache 占用更多显存）
- 无 KV cache 共享（重复 prefix 无法复用）
- 吞吐量较低，但延迟可接受

---

## 五、风险与应对

| 风险 | 概率 | 影响 | 应对措施 |
|------|------|------|---------|
| PyTorch 2.8.0 在 CANN 8.5.0 上安装失败 | 中 | 高 | 使用华为 Ascend pytorch 预编译包；或从源码编译 |
| 910B3 不被 vLLM-Ascend 识别 | 中 | 高 | Patch 设备识别逻辑；设置 SOC_VERSION 环境变量 |
| transformers 5.x 与 vLLM 0.13.0 不兼容 | 高 | 中 | Fork vLLM 放开版本约束；或使用 --trust-remote-code |
| PR #8081 代码与 v0.13.0 API 差异过大 | 高 | 中 | 逐文件对比适配；参考 v0.13.0 已有的 Gemma3 实现 |
| SDPA 回退性能太差 | 中 | 低 | 接受性能折衷；或尝试 head splitting（256→2x128） |
| 单卡 64GB 显存不足 | 低 | 高 | 减少 max_model_len；使用量化（W8A8） |
| MoE 算子在 torch_npu 2.8.0 上不可用 | 中 | 高 | 检查 npu_moe_init_routing_v2 API；回退到 CPU MoE |

---

## 六、参考资源

- vLLM-Ascend 仓库：https://github.com/vllm-project/vllm-ascend
- PR #8081：https://github.com/vllm-project/vllm-ascend/pull/8081
- Issue #8120（Gemma4 模型请求）：https://github.com/vllm-project/vllm-ascend/issues/8120
- PR #8082（SDPA 回退，已关闭）：https://github.com/vllm-project/vllm-ascend/pull/8082
- omni-infer 仓库：https://gitee.com/omniai/omniinfer
- Gemma 4 ModelScope：https://modelscope.cn/models/google/gemma-4-26B-A4B-it
- Ascend PyTorch：https://github.com/Ascend/pytorch
- CANN 兼容矩阵：vLLM-Ascend README

---

## 七、时间线

| 阶段 | 内容 | 预计时间 | 依赖 |
|------|------|---------|------|
| 1.1 | 升级 PyTorch/torch_npu | 0.5 天 | 华为 wheel 包可用性 |
| 1.2 | 安装 vLLM-Ascend | 0.5 天 | 阶段 1.1 完成 |
| 1.3 | 基础推理验证 | 0.5 天 | 阶段 1.2 完成 |
| 2.1 | 准备源码和 PR patch | 0.5 天 | 阶段 1.3 通过 |
| 2.2 | 逐文件适配（10 个文件） | 2-3 天 | 阶段 2.1 完成 |
| 2.3 | transformers 版本冲突处理 | 0.5 天 | 阶段 2.2 过程中 |
| 2.4 | Bug 修复 | 0.5 天 | 阶段 2.2 过程中 |
| 3.1 | 文本推理验证 | 1 天 | 阶段 2 完成 |
| 3.2 | 多模态验证 | 1 天 | 阶段 3.1 通过 |
| 3.3 | 性能基准测试 | 0.5 天 | 阶段 3.2 通过 |
| **总计** | | **7-10 天** | |

如果阶段 1-3 失败，阶段 4（备选方案）可在 **1-2 天** 内完成。
