# CLAUDE.md — vLLM-Omni 贡献工作区

## 这个仓库是什么
vLLM-Omni（`vllm-project/vllm-omni`）是一个**构建在 vLLM 之上**的多模态推理服务框架。
它通过一个**多级引擎**（Orchestrator → StagePool → 各 stage 的 runner，例如 Qwen3-Omni 的
Thinker → Talker → Code2Wav）来服务各类 "omni" 模型（TTS、图像/视频 diffusion、omni 多模态对话）。
它**不是**上游 vLLM。omni / diffusion / MoT 相关代码都在 `vllm_omni/` 下。

## 我是谁 / 目标
- 性能工程师：CUDA kernel、MoT / grouped-GEMM、attention、分布式系统、nsys profiling。
- 硬件：**1× NVIDIA H200（141 GB）**，CUDA。
- 目标：给 vLLM-Omni 做**第一个干净的贡献**，从**跑 test / benchmark** 入手，把测量结果变成可合并的
  产物（tuned config、benchmark / 诊断数据）；只有当 roofline 判断确有 gap 时，才进一步做性能优化。
- 按 `PLAN.md` 的分阶段计划执行，一次只做一个阶段。

## 硬性规则 — 不得违反
1. **未经我明确批准，不得做任何 GitHub 公开动作。** 绝不自行 `git push`、开 PR、开 issue、或在
   GitHub 上发评论。只在本地把 branch / commit / PR 描述**准备成草稿**，然后**停下来给我审**。
2. **提出任何贡献前，先核实这件事没人在做。**"没有 assignee"远远不够。起草任何 PR/issue 前：
   **按内容和代码符号搜 open 和 closed 的 PR**（不只是正式关联的）；若对应某个 issue，去读那个 issue 的
   **评论 + timeline 交叉引用（"mentioned in PR #X"）+ 作者本人开的其它 PR**。若已经有人在做，
   告诉我并停下。（这个项目节奏很快，大量工作是通过"没正式关联的 PR"完成的，且常常是 issue 作者自己在做。）
3. **每个 commit 都签 DCO：`git commit -s`。** 提交前先从 `CONTRIBUTING.md` 确认准确的 sign-off
   与 PR 标题 tag 规范。
4. **绝不把硬件/模型专属的值 hardcode 进共享代码路径。** tuned 结果应放进数据/配置文件（例如
   `configs/` 目录），绝不烧进 kernel 或写成固定常量。任何 fast path 都必须保持可配置。
5. **下"有 gap"结论前先做 roofline。** 说某 kernel "慢"之前，先用 roofline 估算判断这是实现上的 gap
   还是 workload 的物理上限。只追真正的 gap。
6. 遵守 NVIDIA 的开源贡献政策处理身份/署名；使用我配置好的 git 身份。

## 工作方式
- **先读仓库自己的文档**（`CONTRIBUTING.md`、`README.md`，以及运行任何脚本前先读它的 docstring）。
  这些的权威性高于 `PLAN.md` 里的命令；若有出入，以仓库为准并告诉我。
- 所有日志、nsys trace、JSON 结果、临时文件都放在 `./bench_artifacts/` 下（加进本地 `.gitignore`；
  **绝不提交**）。
- 维护一份 `bench_artifacts/RUNLOG.md`：准确命令 → 结果 → 决策。
- 到 `PLAN.md` 的每个 **GATE**：停下、总结（跑了什么、关键数字、拟提交的产物），等我 go/no-go。
- 一个 PR 只做一件聚焦的事。优先最小可评审单元。

## 已核实的仓库路径（用前在 HEAD 复查一遍——仓库变动很快）
- Qwen3-Omni 示例：`examples/offline_inference/qwen3_omni/`、`examples/online_serving/qwen3_omni/`
- 现有 recipe：`recipes/Qwen/Qwen3-Omni.md`
- MoT GEMM 调优器：`benchmarks/kernels/mot_linear_benchmarks.py`
- MoT config 目录（当前**为空** → 机会所在）：`vllm_omni/diffusion/layers/mot/configs/`
- Attention-backend 诊断：`benchmarks/diffusion/bench_attention_backends.py`
