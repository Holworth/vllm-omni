# vLLM-Omni 代码阅读指引

> 基于 commit `68a99206`(2026-07)整理;vLLM 基座为 v0.25.0(editable 安装于 `../vllm-v0.25.0`,
> 跳转可直接进入)。所有 `file:line` 引用均经过实际核对。
> 本机(macOS)可跑 L1 CPU 测试:`pytest -m 'core_model and cpu'`;跑真模型请去 Linux/GPU 机器。

---

## 1. 一句话定位

vLLM-Omni 把 vLLM(文本自回归引擎)扩展为**多阶段(Stage)全模态推理引擎**:
一个模型 = 一条线性 stage 流水线(如 Qwen3-Omni 的 thinker → talker → code2wav),
每个 stage 是一个独立引擎子进程——**AR stage 复用 vLLM 的 EngineCore**,
**Diffusion stage 用自研的 DiffusionEngine**;阶段之间由 Orchestrator 路由控制流、
由 OmniConnector 传输大张量(hidden states / codec codes / KV cache)。

三类模型形态(见 `docs/design/architecture_overview.md`):
- **DiT 为主,AR 做文本编码**:Qwen-Image(单 stage diffusion)
- **AR 为主,DiT 做生成器**:BAGEL(AR stage + diffusion stage,传 KV cache)
- **AR+DiT 全模态**:Qwen3-Omni(3 个 AR/generation stage,传 hidden states)

## 2. 运行时心智模型(先背下这张图)

```
调用方线程                        Orchestrator 线程 (daemon, 自建 asyncio loop)
─────────────                    ──────────────────────────────────────────────
Omni / AsyncOmni                  Orchestrator
  └─ OmniBase                       ├─ _request_handler      (收请求 → stage-0)
       └─ AsyncOmniEngine  ══janus══┤─ _orchestration_loop   (轮询各 stage 输出并路由)
          (3 个 janus 队列:          │
           request / output / rpc)   ├─ StagePool[0]  ── ZMQ ──▶ 子进程: StageEngineCoreProc
                                     │   (每个逻辑 stage 一个池,     (vLLM EngineCore + Omni调度器
                                     │    含 N 个 replica)            + Omni worker/runner)
                                     ├─ StagePool[1]  ── ZMQ ──▶ 子进程: StageEngineCoreProc
                                     └─ StagePool[2]  ── ZMQ ──▶ 子进程: StageDiffusionProc
                                                                   └─ DiffusionEngine
                                                                      └─ MultiprocDiffusionExecutor
                                                                         └─ 每 GPU 一个 WorkerProc
```

**数据流走两条通道**(理解全仓库最重要的一点):
1. **控制面**:Orchestrator 把上游 stage 的输出转成下游的输入请求
   (`_forward_to_next_stage`, `vllm_omni/engine/orchestrator.py:1178`)。对 LLM stage
   常常只发**占位 token**(`[0]*N`,为了让下游调度器预留 KV 槽位)。
2. **数据面**:真正的大张量(hidden states、embeddings、codec codes)走 **OmniConnector**
   (默认 SharedMemoryConnector,跨机用 Mooncake/Mori RDMA),
   以 `OmniPayloadStruct` 形式在 worker 之间直接传输。
   只看 orchestrator 会误以为「没有张量在传」。

两种阶段衔接模式:
- **同步模式**(async_chunk=false):stage 完全跑完 → orchestrator 转发完整 payload。
- **async_chunk 流式模式**:下游 stage 在请求提交时就被"预热"
  (`_prewarm_async_chunk_stages`, orchestrator.py:1464),之后数据按 chunk
  经 `OmniChunkTransferAdapter` 直接 worker→worker 流动,orchestrator 不再参与转发。

## 3. 核心词汇表

| 概念 | 是什么 | 定义处 |
|---|---|---|
| **PipelineConfig / StagePipelineConfig** | 模型的**拓扑**(几个 stage、各自引擎类型、衔接函数),Python 冻结数据类,不可由用户改 | `vllm_omni/config/stage_config.py:242 / :208` |
| **DeployConfig(deploy YAML)** | 模型的**部署**(devices、TP、显存、采样默认值、connector 接线),用户可改 | `vllm_omni/deploy/<model_type>.yaml`;schema 在 `stage_config.py:406` |
| **StageConfig** | 上面两者 + CLI 覆盖的合并产物,每 stage 一个 | `stage_config.py:913`;合并逻辑 `merge_pipeline_deploy:831` |
| **StageExecutionType** | `llm_ar`(自回归)/ `llm_generation`(一次性 vocoder)/ `diffusion` | `stage_config.py:173`;决定调度器 `_resolve_scheduler:181` |
| **OMNI_PIPELINES** | model_type → PipelineConfig 注册表 | `vllm_omni/config/pipeline_registry.py:96` |
| **Orchestrator** | stage 间路由的大脑(orchestrator 线程内 asyncio) | `vllm_omni/engine/orchestrator.py:203` |
| **StagePool** | 一个逻辑 stage 的 replica 集合 + 选路 + 准入 | `vllm_omni/engine/stage_pool.py:48` |
| **OmniConnector** | stage 间大张量传输抽象(put/get 按 (from,to,key)) | `vllm_omni/distributed/omni_connectors/connectors/base.py:12` |
| **OmniPayload / OmniPayloadStruct** | 跨 stage 载荷 schema(hidden_states/embed/ids/codes/meta) | `vllm_omni/data_entry_keys.py:87 / :180` |
| **additional_information** | 随请求跨进程携带的自由张量字典(payload 的 msgspec 线格式) | `vllm_omni/engine/__init__.py:51` |
| **OmniRequestOutput** | 用户拿到的统一输出(text 在 `.request_output`,audio 在 `multimodal_output` 属性,image 在 `.images`) | `vllm_omni/outputs/__init__.py:63` |
| **sampling_params_list** | 每 stage 一个采样参数,长度必须等于 stage 数 | 校验在 `omni_base.py:264` |
| **OmniCoordinator** | ⚠️ 名字有误导——它**不负责串联 stage**,只是分布式模式下 replica 的成员管理/心跳/负载均衡控制面 | `vllm_omni/distributed/omni_coordinator/` |

## 4. 五条关键执行流

### A. 离线:`Omni(model=...)` + `generate()`

构造(一次):
1. `Omni.__init__` → `OmniBase.__init__`(`omni_base.py:131`)→ 建 `AsyncOmniEngine`(`:173`)。
2. AsyncOmniEngine 解析 stage 配置:`StageConfigFactory.create_from_model`
   (`config_factory.py:274`)= OMNI_PIPELINES 注册表 + deploy YAML + CLI 覆盖;
   兜底:legacy `stage_args` YAML(`entrypoints/utils.py:394`)或合成单 stage diffusion 配置。
3. 起 orchestrator 守护线程(`async_omni_engine.py:322`),线程内
   `StageRuntime.initialize`(`stage_runtime.py:230`)为每个 stage replica 拉起子进程:
   LLM → `StageEngineCoreProc`(vLLM EngineCoreProc 子类,`stage_engine_core_proc.py:46`,入口 `run_stage_core:55`);
   Diffusion → `StageDiffusionProc`(`diffusion/stage_diffusion_proc.py:55`)。组装 StagePool。

请求(每条):
4. `generate()` 校验 sampling_params_list;非 diffusion stage 强制 FINAL_ONLY(`omni.py:28`)。
5. `engine.add_request`:**在调用方线程里**做 stage-0 的 tokenize/多模态处理
   (`async_omni_engine.py:714-770`),包成 `StageSubmissionMessage` 入 janus 队列。
6. Orchestrator `_handle_add_request`(`orchestrator.py:435`)→ `StagePool.submit_initial`
   (`stage_pool.py:932`)→ ZMQ 发给 stage-0 子进程。
7. `_orchestration_loop`(`orchestrator.py:659`)轮询所有 replica;LLM 原始输出经
   `MultimodalOutputProcessor.process_outputs`(`outputs/output_processor.py:478`)
   逐步累积多模态张量;`_route_output`(`:876`):final_output stage → 发 OutputMessage 给客户端;
   非终点 stage 完成 → `_forward_to_next_stage`(`:1178`)。
8. 调用方线程 `try_get_output()` 轮询,`_process_single_result`(`omni_base.py:468`)
   包成 `OmniRequestOutput` 返回。请求只有当**所有** final_output_stage_ids 都产出终态才算 finished
   (`orchestrator.py:898-901`;由 prompt 的 `modalities` 键决定,`utils.py:643`)。

### B. 在线:`vllm serve <model> --omni`

- `--omni` 是**两边 CLI 用裸字符串 `"--omni" in sys.argv` 检测的**:上游 vLLM 0.25.0 自带钩子
  (`vllm-v0.25.0/vllm/entrypoints/cli/main.py:42-55`)委托给 vllm_omni;
  `vllm-omni` 脚本反向兜底(`vllm_omni/entrypoints/cli/main.py:12`)。
- `OmniServeCommand`(`cli/serve.py:77`)继承 vLLM 全部 serve 参数(`:186`)再加 ~60 个 omni 参数。
- FastAPI 组装(`openai/api_server.py:460`):**复用** vLLM 的 `build_app/setup_server/serve_http`,
  只摘掉上游 `/v1/chat/completions`、`/v1/models` 换成 omni 路由(`:503-505`);
  chat 用 `OmniOpenAIServingChat`,其余 handler 直接用上游类——因为 `AsyncOmni` 实现了
  vLLM 的 `EngineClient` 协议。引擎侧与离线完全同一套(`AsyncOmni` → 同一个 AsyncOmniEngine)。
- 流式输出按 `final_output_type` 分派(`serving_chat.py:1218`):audio → base64 PCM 放
  `delta.content` + 非标准 `modality:"audio"` 字段;image → base64 PNG;
  `finish_reason` 等所有模态完成才发(`:1940-1948`)。
- `sampling_params_list` 不是声明字段,靠 pydantic `extra="allow"` 从 OpenAI SDK 的
  `extra_body` 进来(`serving_chat.py:586`)。

### C. Diffusion 请求生命周期(Qwen-Image 文生图)

1. StagePool → `StageDiffusionClient.add_request_async`(ZMQ PUSH,`stage_diffusion_client.py:331`)
   → 子进程 `StageDiffusionProc` 重建请求(`stage_diffusion_proc.py:151`)→ `engine.step()`。
2. `DiffusionEngine`(`diffusion_engine.py:133`)busy-loop 线程:两种调度模式——
   **request 模式**(默认):`RequestScheduler` 按 `RequestBatchSamplingParamsKey` 同键分组,
   一次 `pipeline.forward()` 跑完整个去噪循环;
   **step 模式**(`step_execution=True`,流式必需):`StepScheduler` 每 tick 推进一步,
   pipeline 需实现 `prepare_encode / denoise_step / step_scheduler / post_decode` 四方法契约
   (`models/interface.py:46`)。
3. `MultiprocDiffusionExecutor`(`executor/multiproc_executor.py:69`)把调度波次
   `collective_rpc` 广播给每 GPU 一个的 WorkerProc(共享内存 MessageQueue),只有 rank 0 回结果;
   >1MB 张量走 POSIX SHM 旁路(`ipc.py`)。
4. Worker 内:`DiffusionModelRunner`(`worker/diffusion_model_runner.py:95`)持有 pipeline
   (**不是** diffusers 的 DiffusionPipeline,而是自研 nn.Module:text encoder/VAE 用 HF 加载,
   DiT 是 vLLM 原生适配层 `QwenImageTransformer2DModel`,`qwen_image_transformer.py:904`)。
   加速按层附加:cache 后端(cache-dit/TeaCache,`cache/selector.py:11`)、torch.compile、
   offload、USP 序列并行(声明式 `_sp_plan` + hooks)、CFG 并行(正负分支分 rank,
   `distributed/cfg_parallel.py:76`)、VAE tile 并行、HSDP。
   进程组顺序 `tp-sp-pp-cfg-dp`(`diffusion/distributed/parallel_state.py:691`)。

### D. AR stage 如何骑在 vLLM 上

集成策略:**有类钩子就 subclass,没有钩子才 monkey-patch**。三层:
1. **注册**:`import vllm_omni` 先执行 `patch.py`(`__init__.py:22`);
   `vllm.general_plugins` entry point(pyproject.toml:144)让 vLLM 在**每个子进程**
   自动调 `register_omni_models_to_vllm`(`engine/arg_utils.py:91`)——worker 子进程只 import vllm
   也能认识 omni 模型。
2. **线格式**:`OmniEngineCoreRequest/Output(s)`(加 `additional_information` /
   `multimodal_output` 通道,`engine/__init__.py`)通过 patch.py 的 sys.modules 全量替换
   (`patch.py:297-314`)+ 进程边界处显式重打(`stage_engine_core_proc.py:121`、
   `stage_engine_core_client.py:170`)注入 vLLM,使上游 EngineCore busy loop 和 msgpack/ZMQ
   透明携带 omni 载荷。
3. **执行**:调度器走 vLLM 自己的 `scheduler_cls` 配置注入(`OmniARScheduler`,
   `core/sched/omni_ar_scheduler.py:50`——新增:跨 stage KV 传输状态机、connector 驱动的
   准入等待、`OmniEngineCoreOutput` 富输出);worker 走 `worker_cls`(`GPUARWorker` →
   `GPUARModelRunner`,`worker/gpu_ar_model_runner.py:289`,由平台类解析,
   `platforms/cuda/platform.py:30`)。runner 的 `execute_model:940` 前向后用
   `extract_multimodal_outputs`(`gpu_model_runner.py:827`)把模型返回的 `OmniOutput`
   拆成文本 hidden states 和多模态张量,分装进 `multimodal_outputs`(给客户端,走 ZMQ)
   和 `inter_stage_outputs`(给下游,走 connector)。

### E. 跨 stage 载荷实例(Qwen3-Omni)

- thinker → talker:layer-0 输入 embeddings + layer-24 hidden states + token ids
  (`stage_input_processors/qwen3_omni.py:434(async_chunk)/:530(full)`);
  控制面同时发占位 tokens(`thinker2talker_token_only:616`)。
- talker → code2wav:RVQ codec 码块(`:679/:757`)。
- **不是 KV cache**!真 KV 块只在声明 `omni_kv_config` 的管线(如 Bagel thinker→DiT)传输,
  由 `OmniKVTransferManager`(`kv_transfer_manager.py:341`)按层抽取;
  它与 PD 分离用的 vLLM 自带 `kv_transfer_config`(Mooncake)是**两套完全独立的机制**。

## 5. 目录地图

| 目录 | 职责 |
|---|---|
| `vllm_omni/entrypoints/` | Omni/AsyncOmni 门面、CLI(serve)、OpenAI API server |
| `vllm_omni/engine/` | AsyncOmniEngine、Orchestrator、StageRuntime/Pool、stage 客户端/子进程、消息类型 |
| `vllm_omni/config/` | stage_config(拓扑+部署 schema)、pipeline_registry、config_factory、OmniModelConfig |
| `vllm_omni/deploy/` | 各模型的默认部署 YAML(新格式) |
| `vllm_omni/core/sched/` | Omni AR/Generation 调度器(vLLM Scheduler 子类) |
| `vllm_omni/worker/` | Omni GPU worker/runner(vLLM 子类;多模态输出抽取、connector 数据面) |
| `vllm_omni/model_executor/models/` | AR/omni 模型库 + 注册表 + 各模型 pipeline.py(拓扑) |
| `vllm_omni/model_executor/stage_input_processors/` | 跨 stage 载荷打包/解包函数(按模型) |
| `vllm_omni/diffusion/` | 自成体系的 diffusion 引擎(engine/sched/executor/worker/models/cache/distributed/...) |
| `vllm_omni/distributed/omni_connectors/` | 跨 stage 传输(SHM/Mooncake/Mori;chunk 适配器;KV 管理器) |
| `vllm_omni/distributed/omni_coordinator/` | 分布式 replica 成员管理控制面(⚠️ 不路由请求) |
| `vllm_omni/inputs/ outputs/ data_entry_keys.py` | 输入 prompt 类型、输出包装、跨 stage payload schema |
| `vllm_omni/patch.py` | 对 vLLM 的全部 monkey-patch(每条有 WHY/SCOPE/FRAGILITY 注释,必读) |
| `vllm_omni/platforms/` | 平台抽象(cuda/rocm/npu/xpu/musa):选择 worker 类、默认配置路径 |
| `vllm_omni/experimental/` | ar_diffusion、全双工语音等实验特性 |

## 6. 推荐阅读路线

**先跑再读**(macOS 上只能读;跑去 GPU 机器):
`docs/getting_started/quickstart.md` 的 8 行 hello-world;
最佳多阶段首例:`examples/offline_inference/qwen2_5_omni/end2end.py`(594 行,读完整个);
最简 diffusion 例:`examples/offline_inference/text_to_image/text_to_image.py`(main 从 :353)。

### 阶段 1:入口与数据类型(~半天)
1. `entrypoints/omni.py`(215 行,整个同步生命周期)
2. `engine/messages.py`(队列消息词汇表)
3. `outputs/__init__.py`(OmniRequestOutput——用户拿到什么)
4. `inputs/data.py`(Omni prompt 类型 + OmniDiffusionSamplingParams)
5. `data_entry_keys.py`(跨 stage payload 契约)

*自测:sampling_params_list 的长度约束是什么?audio 输出从哪个属性取?一个请求何时算 finished?*

### 阶段 2:编排层(~1 天)
6. `entrypoints/omni_base.py` → 7. `engine/async_omni_engine.py`(:191-450)
8. `engine/stage_runtime.py` → 9. `engine/stage_pool.py`
10. `engine/orchestrator.py`(重点 `_orchestration_loop/_route_output/_forward_to_next_stage`)

*自测:进程/线程拓扑画得出来吗?stage-0 的 tokenize 在哪个线程?占位 token 是干嘛的?*

### 阶段 3:配置系统(~半天)
11. `deploy/qwen3_omni_moe.yaml`(具体例子)→ 12. `model_executor/models/qwen3_omni/pipeline.py`
13. `config/stage_config.py`(schema + merge)→ 14. `config/pipeline_registry.py` + `config/config_factory.py`

*自测:拓扑与部署为何分离?legacy stage_args YAML 和新 deploy YAML 怎么区分?*

### 阶段 4:AR 路径(~1-2 天,建议对照 vllm-v0.25.0 源码读)
15. `patch.py`(全部 patch 目录)→ 16. `engine/__init__.py` + `request.py`(线类型)
17. `engine/arg_utils.py`(注册 + 三桶 CLI 路由,:374 有图)
18. `core/sched/output.py` → `omni_scheduler_mixin.py` → `omni_ar_scheduler.py`
19. `worker/gpu_model_runner.py`(`extract_multimodal_outputs:827`)→ `worker/gpu_ar_model_runner.py`
    (`execute_model:940`、输出组装 `:1673`)
20. `engine/stage_engine_core_proc.py` + `stage_engine_core_client.py`(进程边界重打 patch)

*自测:为什么 patch 要在进程边界重打?多模态张量走哪两条通道出引擎?KV 块何时才释放?*

### 阶段 5:Diffusion 路径(~1-2 天)
21. `docs/design/feature/diffusion_request_level_batching.md` + `diffusion_step_execution.md`(与代码核对过,准确)
22. `diffusion/request.py` → `diffusion/data.py`(OmniDiffusionConfig:573)
23. `diffusion/sched/`(interface → base → request/step scheduler)
24. `diffusion/diffusion_engine.py` → `executor/multiproc_executor.py` → `worker/diffusion_worker.py` → `worker/diffusion_model_runner.py`
25. `models/qwen_image/pipeline_qwen_image.py`(先 forward:996,再四方法)+ `qwen_image_transformer.py`(:904-975 的类属性)
26. 按需:`distributed/cfg_parallel.py`、`hooks/sequence_parallel.py`、`cache/`、`model_loader/diffusers_loader.py`

*自测:request 模式与 step 模式差在哪?为什么流式强制 step 模式且禁 cache?rank 0 之外的 worker 为什么也要执行 RPC?*

### 阶段 6:模型接入(需要加模型时)
27. `model_executor/models/registry.py`(`_OMNI_MODELS:8`;`OmniModelConfig.registry`(`config/model.py:144`)是换掉 vLLM 注册表的唯一钩子)
28. `qwen3_omni/qwen3_omni.py`(一个 wrapper 类按 `model_stage` 分派三个子模型;新模型模板)
29. `stage_input_processors/qwen3_omni.py` + `diffusion/registry.py`
30. 官方教程:`docs/contributing/model/adding_omni_model.md`;仓库自带 skill:`add-diffusion-model` / `add-tts-model`

## 7. 高价值陷阱清单(精选)

1. **patch.py 的 sys.modules 替换只影响已 import 的 vllm 模块**;子进程和晚 import 必须显式重打(`stage_engine_core_proc.py:116`、`stage_engine_core_client.py:164`)。读 vLLM 源码时记住你看到的类可能已被换掉。
2. **两套 YAML 并存**:`model_executor/stage_configs/*.yaml` 是 legacy `stage_args` 格式(如 step_audio_2,别拿它当新模型模板);新格式在 `vllm_omni/deploy/*.yaml` + Python pipeline.py。
3. **`pooler_output` 名不副实**:runner 内部叫 pooler 的 payload 字典,上线时 `OmniModelRunnerOutput.pooler_output` 恒为 None,真通道是 `multimodal_outputs`/`inter_stage_outputs`(`gpu_ar_model_runner.py:1739`)。
4. **给 diffusion stage 传普通 SamplingParams 会被静默丢弃**,换成默认 `OmniDiffusionSamplingParams()`(`stage_pool.py:947`)。
5. **同名类三处**:`OmniRequestState`(outputs)/`OrchestratorRequestState`/`ClientRequestState` 是三层各自的请求状态;diffusion 里还有两个不同的 `DiffusionRequestState`(sched vs worker)。
6. **payload 有嵌套/点号两种形态**(`data_entry_keys.py` flatten/unflatten),混用会静默丢数据;`additional_information` 的 list 里不能放 tensor(会被静默丢弃)。
7. **故障是 fail-stop**:任一 replica 死 → 整个 orchestrator 关停;`Omni.generate` 异常也会 `close()` 整个实例,不可复用(`omni.py:91`)。
8. 使用 deploy YAML 时,**顶层 EngineArgs kwargs 会被剥掉只留警告**(`async_omni_engine.py:1096`)——per-stage YAML 优先。
9. **stage 拓扑严格线性**:`stage_id` 必须等于列表下标,转发永远去 `stage_id+1`(`stage_runtime.py:344`、`orchestrator.py:1190`)。
10. **`_sp_plan` 即 diffusers 的 `_cp_plan` 改名**;SP hooks 由 `registry.initialize_model` 统一施加,模型文件自己不管;没有 `_sp_plan` 的 transformer 静默不做 SP(仅警告)。
11. **衔接函数按点号字符串懒加载**(pipeline.py 里写 dotted path),写错只在运行时爆;签名还会被 `inspect` 探测决定传参(`stage_engine_core_client.py:396`、`orchestrator.py:1216`)——改参数名/数量会改变行为。
12. 加 connector 喂数据的模型要同时改**两个 allowlist**:`_OMNI_CONNECTOR_INIT_ARCHS`(`gpu_ar_model_runner.py:307`)和 `_FULL_PAYLOAD_INPUT_STAGES`(`omni_scheduling_coordinator.py:35`),漏第二个 = 下游静默挂起。
13. 架构名可能**双注册双语义**:`HunyuanImage3ForCausalMM` 在 AR 注册表和 diffusion 注册表各有一份,按 stage 的 execution_type 决定解析到哪个。
14. `import vllm_omni` 的 vLLM 全局注册表**跳过 vLLM 已有的架构名**(`arg_utils.py:98`)——上游同名实现会静默获胜;omni 版本靠 `OmniModelConfig.registry` 生效。
15. 测试标记 `-m` 与 `--run-level` 正交:`-m` 选测试,`--run-level` 决定 fixture 加载 tiny 还是全量权重;nightly 会出现 `-m full_model` + `--run-level core_model` 的组合。

## 8. 设计文档优先级

1. `docs/design/module/async_omni_architecture.md` — 最好的运行时地图(必读)
2. `docs/design/architecture_overview.md` — 动机与三类模型形态
3. `docs/configuration/stage_configs.md` — deploy YAML schema
4. `docs/design/feature/diffusion_request_level_batching.md` + `diffusion_step_execution.md` — 与代码核对准确
5. `docs/contributing/ci/CI_5levels.md` — L1-L5 测试金字塔
- ⚠️ `docs/design/module/entrypoint_module.md` 是一行占位符,别看。

## 9. 测试与本机开发循环

- 层级:`core_model`(L1/L2,每 PR)> `advanced_model`(L3,每 merge)> `full_model`(L4,nightly);标记声明在 `pyproject.toml:235`。
- **本机(无 GPU)可跑**:`pytest -m 'core_model and cpu'`(已验证 `tests/test_outputs.py` 11 passed)。
  CPU 测试集中在 `tests/diffusion`、`tests/model_executor`、`tests/entrypoints`、`tests/engine`。
- 惯用法:`pytestmark = [pytest.mark.core_model, pytest.mark.cpu]`(`tests/test_outputs.py:11`)。
- 本地复刻 CI:`tools/run_ready_jobs.sh`(解析 `.buildkite/test-ready.yml` 逐 job 重放)。
- E2E fixture:`omni_server`(起真实 `vllm serve --omni`)/ `omni_runner`(进程内离线),在 `tests/helpers/fixtures/runtime.py`。

## 10. 进阶话题(首轮可跳过,列出锚点备查)

- **CUDA graph 在 AR stage 的生命周期**:`worker/gpu_ar_model_runner.py`(CUDAGraphMode:19、cudagraph_stats:282、graph 清理:527)——TTS decode 性能关键。
- **多机 headless 部署**:`vllm serve --omni --headless` → `run_headless`(`cli/serve.py:105/:752`),replica 注册到 OmniCoordinator。
- **abort/取消链路**:`Omni.abort`(omni.py:209)→ engine(:1447)→ `Orchestrator._handle_abort`(:408)→ `StagePool.abort_requests`(stage_pool.py:1130)。
- **sleep/wake(RL 权重同步)**:api_server 睡眠引擎集合(~:3398)+ CuMemAllocator patch(patch.py:444)。
- **平台抽象**:`platforms/interface.py` 的 `get_omni_ar_worker_cls` 等钩子,NPU runner 为一等公民(另见 `vllm-omni-npu-upgrade` skill)。
