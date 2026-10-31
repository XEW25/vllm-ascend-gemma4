# deploy/ — Gemma4/Qwen3.5 部署资产与复活指南

本目录收录仓库之外的部署层文件（2026-10-31 从 `/mnt/workspace` 根目录收编），
配合 `gemma4` 分支源码构成完整的可复活单元。

## 文件清单

| 文件 | 内容 |
|---|---|
| `gemma4_npu_deployment_plan.md` | Gemma4 NPU 适配的完整设计计划（2026-04 版；**部分落地**——计划中的 `patch_gemma4_transformers.py` 等新文件未创建，实际本地差异只有 `gemma4` 分支上的 3 个文件改动，见下） |
| `run_qwen35_35b.sh` | Qwen3.5-35B-A3B 生产启动脚本（**Docker 路线**：`quay.io/ascend/vllm-ascend:v0.20.2rc1-a3`，TP=2 × Ascend 910B3，挂载 `/mnt/workspace/models`） |
| `test_qwen35_api.sh` | 服务起好后的 API 冒烟测试（curl 三连） |

## 两层部署结构（复原时别搞混）

1. **开发/调试层**：本仓库源码 + editable 安装进本地 venv
   - 验证过的版本组合：Python 3.12 + vllm 0.18.0 + torch_npu（2.9.0，经 CMakeLists 补丁兼容 2.10.0）+ triton_ascend 3.2.1
   - 基线 commit `99e1ea0f`（= 上游 v0.18.0rc1）+ 本地 3 文件改动（torch 2.10 编译放行、layernorm 算子回退链、triton_utils 适配）
   - 历史构建版本号：`0.18.0rc2.dev0+g99e1ea0fe.d20260604`
2. **生产/服务层**：Docker 镜像自带 vllm-ascend（v0.20.2rc1-a3），**不依赖**本地 editable 安装

## 复活步骤（开发层）

```bash
git clone -b gemma4 git@github.com:XEW25/vllm-ascend-gemma4.git
cd vllm-ascend-gemma4
python3.12 -m venv .venv && source .venv/bin/activate
pip install vllm==0.18.0 triton_ascend==3.2.1   # torch/torch_npu 按 CANN 版本配
pip install -e .                                  # 编译 _C_ascend 算子（需 CANN toolkit）
```

## 复活步骤（服务层）

```bash
# 权重放 /mnt/workspace/models/Qwen/Qwen3.5-35B-A3B（或改脚本里的挂载路径）
bash deploy/run_qwen35_35b.sh                     # docker 起服务，端口 8000
bash deploy/test_qwen35_api.sh                    # 冒烟验证
```

## 其他

- `local-patches/stash-v0.18.0-changes.patch`：历史 stash 备份（与 gemma4 改动重叠、方向不同，未合并，需要时人工比对）
- 当年 editable 安装的 venv 位于 `/mnt/workspace/venv`（若该仓库目录被删，venv 里的 vllm_ascend 随之失效，需按上面步骤重装）
