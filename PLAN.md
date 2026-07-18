# PLAN.md — 给 vLLM-Omni 的第一个贡献（test / benchmark 优先）

**先读 `CLAUDE.md` 并遵守其中的硬性规则** —— 尤其是：未经我批准不得 PR/push/评论；提出任何东西前先核实没人在做。

目标：通过"把它跑起来"熟悉框架，并把每次运行变成一个干净、不撞车的贡献。三个阶段；每个阶段以一个
**GATE** 结束（停 → 汇报 → 等批准后才做任何 GitHub 动作）。

## 交付物追踪（在 `bench_artifacts/RUNLOG.md` 里持续更新）
| # | 产物 | 目标 | 状态 |
|---|------|------|------|
| 1 | Qwen3-Omni H200 recipe 段 + bench 数字 | 小 PR → `recipes/Qwen/Qwen3-Omni.md` | 未开始 |
| 2 | H200 MoT GEMM tuned config + 加速比表 | 新增 `configs/*.json` 的 PR | 未开始 |
| 3 | （拓展）MoT GEMM 性能 gap，nsys + roofline | issue 草稿 | 未开始 |
| 4 | H200 attention-backend 诊断表 | 相关 issue 上的数据 / 小 PR | 未开始 |

---

## Phase 0 — 环境与摸底
- [ ] 确认仓库在 HEAD；`nvidia-smi` 能看到 H200。
- [ ] 按仓库文档安装（源码 `pip install -e .`，或用 `vllm/vllm-omni` docker 镜像 —— **以 README 为准**）。
      验证：`python -c "import vllm_omni, vllm; print('ok')"`。
- [ ] 读 `CONTRIBUTING.md`；记录准确的 **DCO/sign-off**、**PR 标题 tag** 规范，以及任何 CI/lint 步骤。
- [ ] 配好 fork + remote（`origin` = 我的 fork，`upstream` = vllm-project/vllm-omni）。**先别 push。**
- [ ] 读两个 benchmark 脚本的 docstring（路径见 CLAUDE.md）。
- **GATE 0** → 汇报环境状态 + CONTRIBUTING 要点。

## Phase 1 — 跑 Qwen3-Omni（熟悉框架 + 可选 recipe PR）
模型：`Qwen/Qwen3-Omni-30B-A3B-Instruct`（单张 H200 放得下）。
- [ ] 离线单条 prompt：`cd examples/offline_inference/qwen3_omni && bash run_single_prompt.sh` → 确认有音频输出。
- [ ] 离线 TP：`bash run_single_prompt_tp.sh` → 观察 thinker 如何切分。
- [ ] 在线 serving：用 `qwen3_omni_moe_thinking.yaml` 起服务，再 `bash run_curl_multimodal_generation.sh`。
- [ ] RUNLOG 记录：stage 流水线（Thinker/Talker/Code2Wav）、各 stage 的 `max_num_seqs`、启动时间、
      以及一个 latency/throughput 样本。可选：按 recipe 的 benchmark 段跑 `vllm bench serve`。
- [ ] 把 `recipes/Qwen/Qwen3-Omni.md` 和你的运行对照。若它**没有 H200 段**，起草一段（环境、准确启动命令、
      验证命令 + 预期输出、H200 的 `vllm bench` 数字）。**先核实**（CLAUDE.md 规则 2）：是否已有人在给这个
      recipe 加 H200 数字（搜改动该文件的 PR）？
- **GATE 1** → 展示 recipe 段草稿 + 数字；批准后才 PR。

## Phase 2 — MoT GEMM 调优（主贡献）
背景：`vllm_omni/diffusion/layers/mot/configs/` 是空的 → MoT GEMM 目前跑 Triton 默认配置；这个调优器就是
用来生成按硬件的 config 的。干净、且正好在我主场的贡献。
- [ ] **撞车检查**：确认 config 目录仍为空，且搜 open+closed PR 里有没有 MoT-config /
      `mot_linear_benchmarks` 相关工作。若有人在加 H200 config → 停下告诉我。
- [ ] 基线（不 tune）—— 记录默认配置的 latency：
      ```
      python benchmarks/kernels/mot_linear_benchmarks.py \
          --model ByteDance-Seed/BAGEL-7B-MoT --tp-size 1 --dtype w16a16
      ```
- [ ] 调优 + 保存：
      ```
      python benchmarks/kernels/mot_linear_benchmarks.py \
          --model ByteDance-Seed/BAGEL-7B-MoT --tp-size 1 --dtype w16a16 --tune \
          --save-dir vllm_omni/diffusion/layers/mot/configs/
      ```
      可选：对调优器支持的其它 dtype 再跑一遍（如 `w8a8`）。
- [ ] 验证：**不带** `--tune` 重跑一次（让它加载新 config）；做一张**加速比表**（每个 shape：
      默认 µs / tuned µs / 加速比）。
- [ ] Profile + roofline：挑 1–2 个有代表性的 shape；用我的 profiling 方式抓 nsys（tuned vs 默认）；
      对每个做 roofline 分类（compute / memory / launch-bound），记录仍存在的 gap。
- [ ] 交付物 #2：branch + 生成的 `configs/*.json` + PR 描述（做了什么/为什么、调优方法、加速比表、
      HW/SW 版本）。`git commit -s`。
- [ ] （拓展）交付物 #3：若 roofline 显示确有实现上的 gap，**起草**一个带 nsys 证据 + 具体假设的性能
      issue。先别提交。
- **GATE 2** → 展示 config diff + 加速比表 + nsys/roofline 记录（以及任何 issue 草稿）；批准后才 push/PR。

## Phase 3 — Attention-backend 诊断（数据贡献）
背景：`bench_attention_backends.py` 把同一个合成 attention 形状喂给每个 diffusion attention backend 和
SDPA baseline；维护者想要那张显示"某 backend > 1.5× baseline"（可从 auto-route 里关掉的候选）的表。
- [ ] **撞车检查**：搜有没有最近已经在贴 H200 attention-backend 数据的 PR/issue。
- [ ] 运行并保存完整输出（GPU / torch / cuDNN / flashinfer 版本 + 各 backend 的表）：
      ```
      python benchmarks/diffusion/bench_attention_backends.py --preset hv15
      python benchmarks/diffusion/bench_attention_backends.py --preset wan22
      ```
- [ ] 分析：标出在 H200 上 > 1.5× SDPA baseline 的 backend；注明是否 `attn_mask` 路径触发的。
- [ ] 交付物 #4：简洁说明（表 + 结论），作为数据加到相关 issue，或在合适时开个小 PR。
- **GATE 3** → 展示表 + 结论；批准后才发布。

---

## 不要做的事
- 未经我批准，不要开 PR/issue/评论，也不要 push branch（CLAUDE.md 规则 1）。
- 不要把 H200/BAGEL 专属值 hardcode 进共享代码 —— config 是数据文件（规则 4）。
- 没有 before/after 数字**加上** roofline 校验，不要宣称有性能提升（规则 5）。
- 不要提交 `bench_artifacts/` 或大的模型文件。
- 不要因为某 issue 没 assignee 就当它是空的 —— 查 PR/评论/作者（规则 2）。

## 启动
从 **Phase 0** 开始。一次做一个阶段，在每个 **GATE** 停下。
