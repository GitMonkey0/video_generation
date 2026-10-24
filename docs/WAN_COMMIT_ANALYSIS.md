# DiffSynth-Studio Wan 源码说明

## 版本与范围

| 项目 | 内容 |
|---|---|
| 仓库 | `DiffSynth-Studio` |
| Commit | `60b709c4dab8eeabce06ee9bc168328ed4c780e8` |
| Commit 时间 | `2026-10-23 14:45:19 +08:00` |
| Commit 标题 | `Example update (#1779)` |
| 文档日期 | `2026-10-24` |
| 主题 | 当前实现中 Wan 的模型加载、数据处理、训练、推理、权重复用与运行边界 |

本文以该 commit 的本地源码为准，不描述上游其他版本。源码路径以外层仓库根目录为基准，均以 `DiffSynth-Studio/` 开头；命令示例则默认在 DiffSynth-Studio 仓库根目录执行。行号用于定位，函数名和类名用于版本变化后的检索。

主干分析覆盖 Wan2.1 T2V/I2V、Wan2.2 T2V/I2V A14B、Wan2.2 TI2V-5B；VACE、Fun、S2V、Animate、WanToDance 等按其接入主干的位置说明。专项模型的全部内部算子、所有量化后端以及所有功能组合不在逐项执行验证范围内。

VACE、S2V、Animate2 的条件/监督关系，VAE latent 归一化、逐帧表示、量化加载限制及区域编译在后文单独展开。这些内容决定模型所接收数据和训练产物的语义，不只是可选运行参数。

---

## 1. 核心结构

仓库不是分别实现一套 Wan 训练框架和一套 Wan 推理框架，而是复用同一个 `WanVideoPipeline`：

- Pipeline 保存文本编码器、DiT、VAE、图像编码器及可选条件模型。
- PipelineUnit 将外部输入转换为模型所需的中间表示。
- `model_fn_wan_video` 执行模型前向，并按模型类型或输入分派到专项路径。
- 训练模块调用预处理单元和损失函数，不调用完整的推理接口。
- 推理接口调用预处理单元、完整采样循环、后处理和 VAE 解码。
- 训练后的权重重新加载到同一类 Pipeline 的相应模型上。

```text
训练
metadata + 视频/图像/音频
  → UnifiedDataset
  → WanTrainingModule.get_pipeline_inputs
  → PipelineUnitRunner + pipe.units
  → input_latents + context + 可选条件
  → FlowMatchSFTLoss
  → model_fn_wan_video
  → backward + optimizer
  → ModelLogger → safetensors

推理
prompt + 可选条件
  → WanVideoPipeline.__call__
  → PipelineUnitRunner + pipe.units
  → 初始 latents + context + 可选条件
  → timestep 循环
      → model_fn_wan_video
      → CFG
      → scheduler.step
  → post_units
  → VAE decode
  → 帧列表/浮点张量
  → save_video
```

训练与推理的共同点是条件表示和网络计算；差异在于噪声构造、调用次数、损失、梯度和输出处理。

## 2. 源码导航

| 职责 | 入口及关键符号 |
|---|---|
| Wan 训练入口 | `DiffSynth-Studio/examples/wanvideo/model_training/train.py:9`：`WanTrainingModule`、`wan_parser`、主程序 |
| Wan Pipeline | `DiffSynth-Studio/diffsynth/pipelines/wan_video.py:34`：`WanVideoPipeline` |
| 真实模型前向 | `DiffSynth-Studio/diffsynth/pipelines/wan_video.py:1456`：`model_fn_wan_video` |
| 训练公共逻辑 | `DiffSynth-Studio/diffsynth/diffusion/training_module.py:369`：`switch_pipe_to_training_mode` |
| 训练与缓存循环 | `DiffSynth-Studio/diffsynth/diffusion/runner.py:53`：`launch_training_task`、`launch_data_process_task` |
| SFT 和蒸馏损失 | `DiffSynth-Studio/diffsynth/diffusion/loss.py:5`：`FlowMatchSFTLoss`、`DirectDistillLoss` |
| Flow Matching 调度 | `DiffSynth-Studio/diffsynth/diffusion/flow_match.py:47`：`set_timesteps_wan` |
| 保存与日志 | `DiffSynth-Studio/diffsynth/diffusion/logger.py:71`：`ModelLogger` |
| 参数默认值 | `DiffSynth-Studio/diffsynth/diffusion/parsers.py:1` |
| 数据集 | `DiffSynth-Studio/diffsynth/core/data/unified_dataset.py:5`：`UnifiedDataset` |
| 解码、裁剪、取帧 | `DiffSynth-Studio/diffsynth/core/data/operators.py:94`：`ImageCropAndResize`、`LoadVideo` |
| Unit 执行与依赖图 | `DiffSynth-Studio/diffsynth/diffusion/base_pipeline.py:15`：`PipelineUnit`；同文件 `PipelineUnitGraph`、`PipelineUnitRunner` |
| 下载与加载配置 | `DiffSynth-Studio/diffsynth/core/loader/config.py:46`：`ModelConfig` |
| 权重结构识别 | `DiffSynth-Studio/diffsynth/models/model_loader.py:83`：`ModelPool.auto_load_model` |
| 模型注册表 | `DiffSynth-Studio/diffsynth/configs/model_configs.py:1`：`MODEL_CONFIGS` |
| 模型实例化与加载 | `DiffSynth-Studio/diffsynth/core/loader/model.py:1` |
| 文件读取与 hash | `DiffSynth-Studio/diffsynth/core/loader/file.py:1` |
| DiT | `DiffSynth-Studio/diffsynth/models/wan_video_dit.py:338`：`WanModel` |
| VAE | `DiffSynth-Studio/diffsynth/models/wan_video_vae.py:1058`：`WanVideoVAE`；同文件 `WanVideoVAE38` |
| LoRA 格式与融合 | `DiffSynth-Studio/diffsynth/utils/lora/general.py:4`：`GeneralLoRALoader` |
| 显存包装映射 | `DiffSynth-Studio/diffsynth/configs/vram_management_module_maps.py:144` |
| 逐层训练 offload | `DiffSynth-Studio/diffsynth/core/offload_training/manager.py:1` |
| 视频输出 | `DiffSynth-Studio/diffsynth/utils/data/__init__.py:140`：`save_video` |

示例组织：

```text
examples/wanvideo/
  model_inference/                  普通推理
  model_inference_low_vram/         配置显存管理的推理
  acceleration/                    USP 等加速示例
  model_training/
    train.py                       公共 Wan 训练入口
    lora/                          各模型 LoRA 启动脚本
    full/                          各模型全量训练脚本及 Accelerate 配置
    validate_lora/                 加载 LoRA 后生成视频
    validate_full/                 加载全量权重后生成视频
    special/                       缓存训练、FP8、NPU、直接蒸馏等
```

模型清单与示例对应关系位于 `DiffSynth-Studio/docs/zh/Model_Details/Wan.md:58`。清单中的每个链接对应特定模型的脚本，不意味着所有模型可以共用相同权重、条件或采样参数。

## 3. 模型加载：从文件到 Pipeline

### 3.1 ModelConfig 的含义

模型配置可以给出本地 `path`，也可以给出 `model_id` 和 `origin_file_pattern`。本地分片权重用文件列表表示一个模型，而不是把每个分片当成独立模型。

训练 CLI 的两种表示：

```text
--model_id_with_origin_paths
  model_id:file_pattern,model_id:file_pattern,...

--model_paths
  JSON 数组；数组元素可以是单个路径，也可以是表示一个模型的分片路径数组
```

模型标识和文件模式经 `parse_model_configs` 转成 `ModelConfig`。FP8、offload、量化选项也在这一层关联到模型配置。

`--fp8_models` 和 `--offload_models` 的匹配依据是传入的模型配置字符串或路径，不是 Pipeline 中的 `dit`、`vae` 等属性名。量化配置同样按对应模型配置匹配；多文件模型的量化键使用与 `--model_paths` 元素一致的 JSON 列表形式。

### 3.2 下载与文件重定向

默认下载源是 ModelScope。相关环境变量包括：

| 变量 | 含义 |
|---|---|
| `DIFFSYNTH_DOWNLOAD_SOURCE` | 选择 ModelScope 或 HuggingFace |
| `DIFFSYNTH_MODEL_BASE_PATH` | 模型文件基础目录；默认基础目录为 `./models` |
| `DIFFSYNTH_SKIP_DOWNLOAD` | 跳过下载，按配置定位已有文件 |
| `MODELSCOPE_ENDPOINT` | ModelScope 下载端点 |

`WanVideoPipeline.from_pretrained` 默认 `redirect_common_files=True`，把公共 T5、CLIP、Wan2.1/Wan2.2 VAE 的特定 `.pth` 文件模式重定向到 `DiffSynth-Studio/Wan-Series-Converted-Safetensors` 中的对应文件。

因此，脚本中写的原始 model ID 不一定等于公共编码器最终实际读取的文件目录。重定向发生在模型下载和加载之前，直接修改对应 `ModelConfig`。

显式给出本地 `path` 的配置不参与这种基于 model ID 和文件模式的重定向。离线加载仍需要准备 tokenizer，以及专项模型使用的 audio processor 等资源。

### 3.3 模型结构识别

加载顺序为：

```text
ModelConfig.download_if_necessary
  → BasePipeline.download_and_load_models
  → ModelPool.auto_load_model
  → hash_model_file
  → 匹配 MODEL_CONFIGS
  → import 模型类与 state_dict_converter
  → load_model
  → 添加到 ModelPool
  → Pipeline 按 model_name 获取模型
```

这里的 hash 基于权重键名及形状，不是文件内容校验和，也不是模型文件名。它用于识别架构，不用于证明参数数值或文件来源相同。

注册配置决定模型类、实例化参数和权重转换器。`state_dict_converter` 负责把外部权重键映射成仓库内部模块的键。改变文件名不能改变架构识别结果；改变权重键或形状则可能使原注册项不再匹配。

一个权重文件可以包含多个注册模型。`auto_load_model` 遍历匹配项，而不是匹配一个之后立即停止。这也是部分专项权重能够从同一组文件提取主干和附加模块的基础。

### 3.4 Pipeline 中的模型槽位

| 属性 | 角色 |
|---|---|
| `text_encoder` | 文本编码器 |
| `tokenizer` | UMT5 tokenizer，默认资源来自 Wan2.1 T2V-1.3B |
| `dit` | 主去噪模型；A14B 推理中为高噪声模型 |
| `dit2` | 第二去噪模型；A14B 推理中为低噪声模型 |
| `vae` | 视频 latent 与像素之间的编码器/解码器 |
| `image_encoder` | 需要 CLIP 图像条件的模型使用 |
| `motion_controller` | motion bucket 条件 |
| `vace`、`vace2` | VACE 控制模型 |
| `vap` | Video-As-Prompt 模块 |
| `animate_adapter` | Animate 条件适配器 |
| `audio_encoder`、`audio_processor` | S2V 音频条件 |

`from_pretrained` 请求最多两个同名 DiT。存在两个时，按 ModelPool 中的加载顺序分配给 `dit`、`dit2`，不是根据路径中的 `high_noise` 或 `low_noise` 自动排序。仓库 A14B 示例先配置 high-noise，再配置 low-noise。

未加载的可选模型槽位可以是 `None`。加载日志中的某个可选模型不存在，不等于当前任务失败；是否必要取决于实际执行分支。

### 3.5 架构参数也是权重契约的一部分

`MODEL_CONFIGS` 除了确定模型类，还确定 `in_dim`、`out_dim`、隐藏维度、层数、patch 及条件开关。相同的 Pipeline 属性名不能说明两份权重具有相同输入语义。

例如 TI2V-5B 配置使用 48 通道 latent，并开启 `fuse_vae_embedding_in_latents`、`seperated_timestep`，关闭常规 VAE/CLIP 条件拼接。S2V 注册为 `WanS2VModel`，而不是普通 `WanModel`；Animate2 注册为 `WanAnimate2Transformer`。

后两者仍被加载到 `pipe.dit`，但实际前向分支和输入条件不同。权重结构能够识别、键名能够加载，并不替代对模型类和条件开关的确认。

依据：`DiffSynth-Studio/diffsynth/configs/model_configs.py:101`、`DiffSynth-Studio/diffsynth/configs/model_configs.py:308`。

## 4. 输入、中间表示与尺寸约定

### 4.1 三个输入字典

预处理与前向以三个字典传递状态：

| 字典 | 典型内容 |
|---|---|
| `inputs_shared` | 尺寸、视频、噪声、latents、图像/音频/控制条件、CFG 配置等 |
| `inputs_posi` | 正向 prompt 及其 `context` |
| `inputs_nega` | 负向 prompt 及其 `context` |

同名键可能被后续 Unit 更新。例如 `latents` 从初始噪声、视频加噪或首帧融合结果开始，在采样循环中持续更新。字典是整个 Pipeline 的运行状态，不是只读配置。

### 4.2 常见张量

| 名称 | 语义和形状 |
|---|---|
| 视频像素 | 常规预处理为 `[B, 3, F, H, W]`，范围归一化到 `[-1, 1]` |
| `input_latents` | 监督视频的干净 latent，常规形状 `[B, C_z, F_z, H_z, W_z]` |
| `noise` | 初始化阶段生成的噪声张量 |
| `latents` | 当前模型输入或当前采样状态 |
| `context` | 文本编码特征，默认 tokenizer 序列长度为 512 |
| `clip_feature` | 需要图像编码器的模型使用的图像特征 |
| `y` | 常规 I2V 等路径使用的 VAE 条件、掩码或控制特征 |
| `first_frame_latents` | TI2V 等首帧融合路径需要固定的图像 latent |
| `vace_context` | VACE 控制条件 |

`noise` 和 `latents` 不是永远相同。标准 SFT 的 loss 会重新生成训练噪声并覆盖 `latents`；推理中的 `latents` 则由采样循环不断更新。

### 4.3 VAE 与 patch 尺寸

| VAE 实现 | 常用模型 | latent 通道 | 空间压缩倍数 |
|---|---|---:|---:|
| `WanVideoVAE` | Wan2.1；Wan2.2 A14B 示例也使用 Wan2.1 VAE | 16 | 8 |
| `WanVideoVAE38` | Wan2.2 TI2V-5B 使用的 Wan2.2 VAE | 48 | 16 |

常规时间长度为：

```text
F_z = (F - 1) // 4 + 1
```

Pipeline 的常规帧数要求为 `F = 4n + 1`，例如 49、81。特殊逐帧处理不能简单套用常规时空压缩约定。

Pipeline 加载 VAE 后，把宽高整除因子设为 `vae.upsampling_factor * 2`。这将 VAE 空间压缩和 DiT 空间 patch 的尺寸约束合在一起：常规 Wan2.1 为 16，TI2V-5B 为 32。

以 Wan2.1 VAE、DiT 空间 patch 为 `2×2`、像素尺寸 `480×832` 为例：

| 视频帧数 | latent 形状 | DiT 视频 token 数 |
|---:|---|---:|
| 49 | `[1, 16, 13, 60, 104]` | `13 × 30 × 52 = 20,280` |
| 81 | `[1, 16, 21, 60, 104]` | `21 × 30 × 52 = 32,760` |

这里没有计入参考帧等专项路径额外加入的 token。帧数、分辨率和 patch 数同时影响 DiT 的序列长度；VAE tiling 不改变这个序列长度。

### 4.4 latent 数值不是原始 encoder 输出

常规 `VideoVAE_.encode` 得到 `mu` 和 `log_var` 后，返回的是按通道归一化的 `mu`：

```text
z = (mu - channel_mean) / channel_std
decode 输入 = z * channel_std + channel_mean
```

`channel_mean` 和 `channel_std` 是 VAE 类中定义的常量，不是当前样本的统计量。Wan2.1 VAE 和 Wan2.2 VAE 使用不同的通道数与常量。

这条 encode 路径没有调用 posterior 随机采样。虽然类中存在 reparameterize/sample 方法，普通 Pipeline 编码目标视频得到的是归一化 posterior mean。将外部未经同样归一化的 latent 直接作为 `input_latents`，不等于复用了相同监督表示。

时间编码先处理第一帧，再以每四帧为一组处理后续帧；因果卷积缓存在一次编码内部沿时间传递，并在新一次底层 encode 开始时清理。decode 也按 latent 时间片顺序使用内部缓存。因此常规视频编码不是逐帧独立图像编码后简单堆叠。

依据：`DiffSynth-Studio/diffsynth/models/wan_video_vae.py:951`、`DiffSynth-Studio/diffsynth/models/wan_video_vae.py:1058`、`DiffSynth-Studio/diffsynth/models/wan_video_vae.py:1370`。

### 4.5 framewise 和 tiling 改变的内容不同

`encode_framewise` 对每个输入像素帧分别调用单帧编码，再沿时间维拼接；`decode_framewise` 对每个 latent 帧分别解码。framewise 改变时间表示，不能视为普通视频 VAE 的低显存等价实现。

WanToDance global 训练还会在共享输入含 `framewise_decoding=True` 时，把 `num_frames` 改为 `4 * (加载帧数 - 1) + 1`，使常规 NoiseInitializer 产生与逐帧 latent 数量对应的时间网格。

VAE 空间 tile size/stride 在接口中按 latent 尺度配置。encode 内部乘以 VAE 空间压缩倍数，转换成像素裁剪尺寸；decode 在 latent 网格上切块。重叠区域按边缘权重融合，而不是逐块直接拼接。tile size 因此不等于输入视频的像素块大小，也不保证与不分块结果数值完全一致。

常规 VAE 的 encode/decode 包装还包含显式 CPU 暂存和逐样本计算；tiled 路径在 CPU 累积块输出。不能仅通过 `pipe.device` 推断中间张量始终驻留 GPU，或浮点视频输出必然位于同一设备。

当前 encode 的多样本循环在每次迭代中直接改写 `tile_size` 和 `tile_stride`，再次乘以空间压缩倍数。同一次 tiled encode 处理第二个及后续样本时，这个转换会累乘。例如压缩倍数为 8、初始 tile size `(2, 3)` 时，前两次传给 tiled_encode 的值分别是 `(16, 24)` 和 `(128, 192)`。这是包装方法的调用行为，不等于这里已量化其对完整 VAE 输出的影响；单样本路径没有跨样本累乘。

依据：`DiffSynth-Studio/diffsynth/models/wan_video_vae.py:1218`、`DiffSynth-Studio/diffsynth/models/wan_video_vae.py:1250`、`DiffSynth-Studio/examples/wanvideo/model_training/train.py:68`。

## 5. 数据读取与训练输入组装

### 5.1 Metadata

`UnifiedDataset` 支持 CSV、JSON、JSONL。普通 Wan SFT 的最小语义是视频与描述：

```csv
video,prompt
videos/sample.mp4,"一只小狗在草地上奔跑"
```

视频路径由 `dataset_base_path` 解析。`dataset_repeat` 扩大每个 epoch 的数据集长度；实际索引通过取模重复访问基础样本。它本身不引入新的视频片段或额外数据增强。

两类参数不能混淆：

- `--data_file_keys` 决定哪些 metadata 字段需要执行文件加载算子；默认 `image,video`。
- `--extra_inputs` 决定哪些字段或派生条件送入 Pipeline。

例如音频字段既需要列入文件加载键，也需要作为额外输入传入 Pipeline；否则可能传入路径字符串，而不是音频波形。

### 5.2 视频采样

当前训练入口使用 `UnifiedDataset.default_video_operator`，其行为是：

1. 根据后缀选择图像、GIF 或视频加载器。
2. 对视频从序列开头取帧。
3. 当原视频帧数不足时减少目标帧数，并调整到时间整除约定。
4. 对每帧等比例缩放，再中心裁剪到目标尺寸。

默认 `fix_frame_rate=False`。虽然算子有 `frame_rate` 参数，当前 Wan 训练入口未启用统一 FPS，实际按原视频帧索引顺序读取。训练入口也没有默认随机起始帧采样。

尺寸处理分两种情况：

- 同时提供 `height`、`width`：使用给定尺寸，不在 `ImageCropAndResize` 中自动对齐。
- 任一尺寸未提供：按原图比例与 `max_pixels` 计算尺寸，再向下对齐整除因子。

当前训练入口给数据算子传入的空间整除因子固定为 16；Pipeline 自身的尺寸检查则按实际 VAE 的约束向上对齐。它们是两个独立步骤。

Pipeline 的尺寸检查更新输入字典中的尺寸，不等于重新裁剪或缩放已经由 Dataset 加载的监督视频。

### 5.3 特殊文件算子

训练入口单独处理：

| 字段 | 行为 |
|---|---|
| `animate_face_video` | 按 `512×512` 处理的面部视频 |
| `input_audio` | 加载为 16 kHz 音频 |
| `wantodance_music_path` | 转为绝对路径，由专项路径继续处理 |

其他文件字段使用默认视频算子，除非另行登记专用算子。不是所有 metadata 列都会自动读取、自动转换或自动传给模型。

### 5.4 WanTrainingModule 的输入映射

普通训练输入包含：

```text
inputs_posi.prompt = data.prompt
inputs_shared.input_video = data.video
height/width = 实际加载视频首帧尺寸
num_frames = 实际加载帧数
cfg_scale = 1
tiled = False
rand_device = pipe.device
max/min_timestep_boundary = 训练模块配置
```

`extra_inputs` 的特殊映射包括：

- `input_image` 从目标视频第一帧取值。
- `end_image` 从目标视频最后一帧取值。
- `reference_image`、`vace_reference_image`、`animate2_reference_image` 从对应已加载序列的第一项取值。
- 其他字段从 `data` 中按原名取值。

因此常规 I2V 训练中的 `input_image` 不是直接读取 metadata 同名字段，而是由目标视频首帧派生。`input_video` 在普通 SFT 中是监督目标，不表示该训练任务天然就是 Video-to-Video。

`extra_inputs` 在基础共享输入组装之后处理，同名字段可以覆盖基础值。直接蒸馏示例就是通过这一机制传入 seed、随机设备、步数等。

### 5.5 类型转换并不是通用 schema 规范化

训练前向中的 `transfer_data_to_device` 递归处理 tensor、tuple、list 和 dict：

- tensor 移到目标设备。
- FP32、FP16、BF16 tensor 转成 Pipeline 指定浮点 dtype。
- FP64、整数和布尔 tensor 不按上述规则转换 dtype。
- NumPy 数组、字符串、Python 标量、PIL 图像等对象保持原类型，由后续 Unit 处理。

因此 metadata 中的布尔值、列表或数字是否被正确解析，取决于 metadata 格式和具体字段处理，不存在一次统一的强类型 schema 转换。例如字符串 `"false"` 不会在这里自动变成布尔 False。

依据：`DiffSynth-Studio/diffsynth/diffusion/training_module.py:133`。

### 5.6 文件路径列表与帧列表不是同一输入契约

`UnifiedDataset.default_image_operator` 分别支持字符串路径和路径列表；Wan 训练主入口使用的 `default_video_operator` 只有字符串路由，再按扩展名读取单张图、GIF 或视频。JSON metadata 能表达列表，不代表 `data_file_keys` 中的视频字段可以直接填写图片路径列表；未匹配的类型由 `RouteByType` 抛出异常。

Pipeline 推理接口接收已加载的 PIL 帧列表，是媒体读取之后的输入契约，不能反推 Dataset 也支持这种 metadata。mask 索引、标量开关等不需要媒体读取的字段，应与文件键区分；是否传入 Pipeline 仍由训练输入映射决定。

依据：`DiffSynth-Studio/diffsynth/core/data/unified_dataset.py:27`、`DiffSynth-Studio/diffsynth/core/data/operators.py:253`。

## 6. PipelineUnit 的执行规则

### 6.1 普通 Unit

普通 Unit 声明 `input_params`、`output_params`，从共享字典取参数，返回字典并更新共享状态。

参数通过 `.get()` 获取，缺失时通常得到 `None`。很多可选 Unit 在条件不存在时返回空字典，从而不改变状态。必要输入是否缺失，最终由具体 Unit 或模型分支决定。

`onload_model_names` 声明该 Unit 使用的模型，也是缓存拆图识别模型依赖的重要信息。加载或卸载动作由 Unit 中的 `load_models_to_device` 配合显存管理执行。

### 6.2 CFG 分离 Unit

源码属性名为 `seperate_cfg`。这类 Unit 分别处理正向和负向输入：

- 正向分支始终执行。
- `cfg_scale != 1` 时，负向分支独立执行。
- `cfg_scale == 1` 时，将正向处理输出同步到负向字典，不额外执行负向编码。

PromptEmbedder 使用这种机制。普通 SFT 的 `cfg_scale=1`，因此不计算独立的负向文本编码。训练入口未实现默认随机空 prompt 或条件丢弃策略。

### 6.3 take_over Unit

`take_over=True` 的 Unit 直接接收三个输入字典，并返回更新后的三个字典，可以管理复杂输入流。它仍然依赖声明的输入、输出和模型关系参与拆图。

### 6.4 当前预处理顺序

`WanVideoPipeline.__init__` 中的 `units` 顺序为：

```text
ShapeChecker → NoiseInitializer → PromptEmbedder → S2V
→ InputVideoEmbedder
→ ImageEmbedderVAE → ImageEmbedderCLIP → ImageEmbedderFused
→ FunControl → FunReference → FunCameraControl → SpeedControl
→ VACE
→ AnimateVideoSplit → AnimatePoseLatents → AnimateFacePixelValues → AnimateInpaint
→ VAP → UnifiedSequenceParallel → TeaCache → CfgMerger
→ LongCatVideo
→ WanToDance_ProcessInputs → WanToDance_RefImageEmbedder
→ WanToDance_ImageKeyframesEmbedder
→ Animate2RefPromptEmbedder → Animate2CLIPEmbedder
→ Animate2VAEEmbedder → Animate2RefKVCacheEmbedder
```

`post_units` 包含 S2V 和 Animate2 后处理，位于采样结束与 VAE 解码之间。普通 SFT 训练前向只遍历 `pipe.units`，不会执行完整推理接口里的后处理和解码。

这个列表不是每个任务都执行有效计算的列表；多数专项 Unit 由是否存在相应条件决定是否生效。

### 6.5 拆图依赖声明与实际执行的边界

`PipelineUnitGraph` 依赖 Unit 声明，不分析真实 tensor 运算或 autograd 图。直接模型依赖使用 `model_name in unit.onload_model_names` 的精确名称匹配。训练冻结逻辑能解析 `dit.blocks.0` 这类子模块路径，不代表拆图时会将其归一化为 `dit`；二者识别粒度不同。

当前旧版 Animate 的 Face Unit 将 `output_params` 写成 `("face_pixel_values")`，实际为字符串而非单元素 tuple。依赖图遍历该声明时得到各个字符，但 take-over 执行仍直接写入完整的 `face_pixel_values` 键。WanToDance ProcessInputs 也在运行时写入 `music_feature` 和负向 `skip_9th_layer`，却未声明输入、输出或模型依赖。

因此声明图不等于完整的数据流证明。专项条件是否被拆入缓存侧或训练侧，需要结合实际字段读写和所选训练模块判断；这些声明差异本身不构成所有拆分配置均失败的执行结论。

依据：`DiffSynth-Studio/diffsynth/diffusion/base_pipeline.py:445`、`DiffSynth-Studio/diffsynth/pipelines/wan_video.py:998`、`DiffSynth-Studio/diffsynth/pipelines/wan_video.py:1075`。

## 7. 条件编码的几种不同路径

### 7.1 文本

PromptEmbedder 使用 tokenizer 生成 token ID 和 mask，经文本编码器得到 `context`。默认 tokenizer 序列长度为 512；超过有效序列长度的位置在文本特征中置零。

之后在 DiT 前向中，`context` 再经过 `dit.text_embedding` 投影到模型隐藏维度。

### 7.2 目标视频与 Video-to-Video

InputVideoEmbedder 先对视频做像素预处理，再用 VAE encode：

- 训练 scheduler 状态下返回 `input_latents`，并以初始化噪声设置 `latents`。
- 推理状态下，将输入视频 latent 按第一个采样 timestep 加噪，作为 `latents`。
- 没有输入视频时，直接以噪声作为初始 `latents`。

这解释了为什么同一个 Unit 能服务训练监督视频与推理 Video-to-Video，但两者的实际计算语义不同。

### 7.3 常规 I2V / 首尾帧

ImageEmbedderVAE 构造条件视频：首帧放入输入图像，其余位置填零；存在 end image 时末帧也放入图像。它编码条件视频，并把时间掩码和 VAE latent 拼成 `y`。

对于常规 Wan2.1 VAE，掩码整理后为 4 个通道，条件 latent 为 16 个通道。是否需要这条路径由 `dit.require_vae_embedding` 控制。

ImageEmbedderCLIP 只在模型需要 CLIP 条件、存在输入图像且具备相应编码器时生效。是否使用 CLIP 由具体模型配置决定，不能把所有 I2V 模型视为同一条件结构。

### 7.4 Wan2.2 TI2V-5B

ImageEmbedderFused 在 `fuse_vae_embedding_in_latents` 启用时，把输入图像的 latent 写入 `latents` 的第一帧，同时保存 `first_frame_latents`。

- 推理每次 scheduler 更新后重新写回首帧。
- SFT 构造带噪 latent 后同样重新写回首帧。
- SFT 的预测与监督目标均排除首帧后再计算损失。
- `seperated_timestep` 路径可以为首帧 token 使用零 timestep，其他视频 token 使用当前噪声 timestep。

这与常规 I2V 的 `y` 拼接路径不同。

### 7.5 控制与专项条件

| 家族 | 接入方式 |
|---|---|
| Fun Control | 控制视频编码为 latent，并构造或组合 `y` |
| Fun Reference | 参考图编码成额外 reference latent 等条件 |
| Camera | 生成相机控制表示并接入 patchify/control adapter |
| VACE | 生成控制特征；在指定 DiT block 加入对应 hint |
| VAP | 维护额外提示视频特征及其网络计算 |
| S2V | 音频特征、运动上下文和可选姿态；分派到专项 S2V 前向 |
| Animate | 姿态 latent、面部像素和适配器注入 |
| Animate2 | 使用专项模型与参考特征/KV 等条件 |
| WanToDance | 音乐、参考图、关键帧及专项时间/FPS 处理 |

专项路径可能在时间维度加入参考帧，或在返回视频前删除附加帧。不能仅按通用 `num_frames` 和普通 latent 公式推断其全部内部形状。

### 7.6 VACE：mask、参考帧与训练目标

VACE 不只是将一个控制视频 encode 后传给 DiT。当前 Unit 先把 mask 转为 `[0, 1]`，构造两路视频：

```text
inactive = video * (1 - mask)
reactive = video * mask
```

两路分别经 VAE 编码，并沿通道维拼接。mask 取第一通道，将每个 `8×8` 空间块整理为 64 通道，再用 nearest-exact 缩小时间长度。在 Wan2.1 VAE 下，最终 `vace_context` 通常为 `16 + 16 + 64 = 96` 通道。

没有提供 mask 时使用全一 mask，所以 inactive 路为零、reactive 路包含控制视频。mask 对应的是两路划分的权重，不能只按其他 inpainting 接口的命名猜测其作用。

参考图按单帧分别编码，并在时间维前置到 VACE 条件中；对应 mask 槽置零。NoiseInitializer 为参考图增加时间槽，训练 InputVideoEmbedder 也把参考图 latent 前置到 `input_latents`。

普通 SFT loss 没有专门排除 VACE 参考槽，所以这些槽参与该路径的预测和监督；推理结束时则删除参考槽，不把它们作为输出视频帧。这里的“训练参与、输出删除”与 TI2V 首帧从 loss 排除的逻辑不同。

当前训练入口默认对 `vace_reference_image` 取已加载序列的第一项，所以其标准输入映射是单参考图。Unit 接受参考图列表，不代表标准训练入口自动保留多参考图。手工传入多参考图时还要区分：VACE 条件侧逐参考图独立编码，InputVideoEmbedder 的目标侧则将参考图序列作为一个视频编码；两者并非同一种时间组织。

VACE Unit 将过长控制视频和 mask 截到 `num_frames`，但没有通用的短视频补齐或几何对齐步骤。条件、mask、目标视频的时间长度与宽高仍需符合该路径的形状要求。

当前 Wan2.1 VACE-1.3B 示例的全量训练目标为 `vace`，LoRA 目标也为 `vace`，不是默认训练 `dit`。冻结 DiT 的权重并不等于通过 DiT 的反向传播被关闭；VACE hints 仍可通过后续网络计算接收梯度。

依据：`DiffSynth-Studio/diffsynth/pipelines/wan_video.py:674`、`DiffSynth-Studio/examples/wanvideo/model_training/full/Wan2.1-VACE-1.3B.sh:1`、`DiffSynth-Studio/examples/wanvideo/model_training/lora/Wan2.1-VACE-1.3B.sh:1`。

### 7.7 S2V：音频分段、首帧和 motion history

S2V Unit 启用要求音频输入或预计算 embedding 存在，同时音频编码器和 processor 已加载。它把正向音频 embedding 放到 `inputs_posi`，负向分支使用零音频 embedding；motion 和 pose 条件放到共享字典。这与纯 T2V 只对文本做 CFG 的条件组织不同。

普通音频处理默认使用 16 FPS，按 `num_frames - 1` 帧分段。音频编码器可返回多段 embedding，但普通 `process_audio` 只使用第一段；`pre_calculate_audio_pose` 才返回全部分段和段数。一次普通 Pipeline 调用因此不是自动生成覆盖任意长音频的全部视频。

`motion_video` 的实际处理需要 `[B, 3, 73, H, W]` 像素张量，不能只按推理接口中的列表类型注解传入 PIL 帧列表。没有 motion history 时构造 73 帧零张量，并设置 `drop_motion_frames=True`；有 history 时设置 False，然后用 VAE 编码。

姿态条件按生成部分的 `num_frames - 1` 截取，短缺时补 `-1` 像素帧。编码前临时增加首帧，编码后去掉对应首个 latent，得到与生成部分对齐的姿态 latent。

`WanS2VModel` 开启首帧 latent 融合。专项前向把首帧作为 reference、其余作为待生成部分，再注入音频、motion 和 pose。普通 SFT 因 `first_frame_latents` 排除首帧监督；这是条件首帧，而不是普通生成视频帧。

多段示例在 Pipeline 外管理连续性：预计算音频/姿态分段，多次调用 Pipeline，获取浮点视频，将前一段尾部像素作为后续 motion history，裁剪重叠部分并拼接输出。音轨合成也由保存函数完成，不由普通 Pipeline 返回。

`save_video_with_audio` 先写视频，再调用 `merge_video_audio`。后者通过 PATH 中的 `ffmpeg` 可执行文件合成音轨，并使用 `-shortest` 截到较短流的长度；这与 imageio 自身的视频写出是两个步骤。合成执行失败时函数捕获异常并打印信息，因此生成 MP4 文件本身不证明音轨已成功合成。

依据：`DiffSynth-Studio/diffsynth/pipelines/wan_video.py:865`、`DiffSynth-Studio/diffsynth/pipelines/wan_video.py:1821`、`DiffSynth-Studio/diffsynth/models/wav2vec.py:186`、`DiffSynth-Studio/examples/wanvideo/model_inference/Wan2.2-S2V-14B_multi_clips.py:1`。

### 7.8 Animate2：训练参考分支与推理 KV 缓存

Animate2 使用独立的 `WanAnimate2Transformer`。参考视频 prompt 生成 `context_ref`；参考图与参考视频首帧分别生成图像特征；VAE Unit 生成参考 latent、生成条件、grid 和序列长度等参数。

存在参考视频时，生成 latent 时间维增加一个参考图槽。训练 InputVideoEmbedder 在满足参考图和参考视频条件时，同样把参考图 latent 加到目标前部；post unit 在推理输出前删除首槽。该槽不是 TI2V 的 `first_frame_latents` 特殊排除路径。

参考 KV Unit 声明依赖 `dit`，但在 `scheduler.training=True` 时不预计算 KV。专项前向按是否存在 KV 字典分为：

- 没有 KV：调用 `dit.forward_origin`，把参考与生成部分放入同一次前向，供训练使用。
- 有 KV：预处理时先 `dit.forward_ref`，迭代时调用 `dit.forward_gen` 复用参考 KV，供推理使用。

因此参考 KV 并不是默认缓存到磁盘后供训练反复复用的固定特征；训练参考分支属于模型计算图的一部分。

现有 Animate2 LoRA 脚本采用显式两阶段：缓存阶段只加载文本、图像编码器和 VAE，训练阶段只加载 Animate2 DiT。其 LoRA 目标为 `self_attn.q,self_attn.k,self_attn.v`，并非普通 Wan 示例的全部 `q,k,v,o,ffn` 投影。

这也说明 Split training 不要求两阶段加载同一套模型文件：真正的约束是缓存表示、保留 Unit 和所需模型吻合。Animate 与 Animate2 的 adapter 路径和 transformer 参考分支不能互换理解。

依据：`DiffSynth-Studio/diffsynth/pipelines/wan_video.py:1293`、`DiffSynth-Studio/diffsynth/pipelines/wan_video.py:1909`、`DiffSynth-Studio/examples/wanvideo/model_training/lora/Wan2.2-Animate-2-14B.sh:1`。

### 7.9 Animate：姿态、面部与参考时间槽

这里的 `Wan2.2-Animate-14B` 使用 `WanAnimateAdapter`，不是上一节 Animate2 的参考 KV 路径。

- 姿态视频经 VAE 编码，再经适配器的 Conv3d patch embedding 加到 `x[:, :, 1:]`，跳过首个参考时间槽。
- 面部条件保留为像素序列，经 motion encoder 和 face encoder 得到特征；前端补一个零参考位置。DiT 的 block 编号满足 `block_idx % 5 == 0` 时加入 face fuser 残差。
- 正向 `face_pixel_values` 来自输入面部视频，负向条件为同形状的全 `-1` tensor，不是零 tensor，也不是另一段面部视频。
- 训练有 `input_video` 时，AnimateVideoSplit 将姿态、面部、背景和 mask 条件各自截到 `len(input_video)-4`，目标视频本身不在这里截断。
- 推理结束时，只要同时传入姿态和面部视频，最终裁剪条件就会移除首个 latent 时间槽。普通 VAE 下，`num_frames=81` 对应 21 个 latent 时间位置，删除一个后按解码公式得到 77 帧；推理示例的条件视频也取 `81-4` 帧。这是源码裁剪和长度公式，不是本环境完整 VAE 生成的实测结果。

AnimateInpaint 对输入 mask 先执行 `1 - preprocess_video(..., min_value=0, max_value=1)`，再 nearest resize 到 latent 空间；不能套用 VACE 的 mask 方向解释。姿态 Unit 也没有通用的自动空间 resize，条件的空间、时间对齐必须满足具体适配器计算。

训练示例目录名称不能代替训练对象判断：full 示例指定 `trainable_models=animate_adapter`、去掉 `pipe.animate_adapter.` 前缀；LoRA 示例在 `dit` 上注入 LoRA、去掉 `pipe.dit.` 前缀。这里的 full 不是全量 DiT 训练。

依据：`DiffSynth-Studio/diffsynth/pipelines/wan_video.py:960`、`DiffSynth-Studio/diffsynth/pipelines/wan_video.py:1014`、`DiffSynth-Studio/diffsynth/pipelines/wan_video.py:354`、`DiffSynth-Studio/diffsynth/models/wan_video_animate_adapter.py:615`、`DiffSynth-Studio/examples/wanvideo/model_training/full/Wan2.2-Animate-14B.sh:1`、`DiffSynth-Studio/examples/wanvideo/model_training/lora/Wan2.2-Animate-14B.sh:1`。

### 7.10 Fun Control、Reference 与 Camera

这三种条件并非同一种 latent 拼接：

| 条件 | 当前数据路径 | 主要约束 |
|---|---|---|
| Fun Control | 控制视频经 VAE 编码，前缀拼入 `y`；剩余通道数为 `dit.in_dim-control_channels-latent_channels` | 若 `clip_feature` 或 `y` 任一缺失，重建零 CLIP `[1,257,1280]` 和零剩余 `y`；已有条件可能被替换，而非逐字段补齐 |
| Fun Reference | 参考图 resize 后编码为 `reference_latents`，可额外生成 CLIP；前向经 `dit.ref_conv` 生成前缀 token | 前向增加时间位置，head 后移除参考 token，再 unpatchify；不同于直接在 sampler latent 中增加参考槽 |
| Camera | 生成六通道 Plücker ray，首帧重复四次后按四帧分组为 `[B,24,F_z,H,W]` | 这不是 VAE 编码结果；control adapter 做 pixel-unshuffle 8 和 stride-2 卷积后加到 patch 特征 |

Camera 分支由 `camera_control_direction` 触发，并直接使用 `dit.control_adapter` 和 `input_image.resize`，不会仅凭一个方向参数补全模型和图像条件；它还重建首图条件 `y`。Fun Control 不通用地补帧或 resize 控制视频。多个条件同时启用时，按 §6.4 的 Unit 顺序读写共享 `y`、`clip_feature`，不能认为各条件独立且自动叠加。

`motion_bucket_id` 则通过 motion controller 改变 `t_mod`，与相机表示或输出视频 FPS 都不是同一机制。

依据：`DiffSynth-Studio/diffsynth/pipelines/wan_video.py:559`、`DiffSynth-Studio/diffsynth/pipelines/wan_video.py:585`、`DiffSynth-Studio/diffsynth/pipelines/wan_video.py:608`、`DiffSynth-Studio/diffsynth/models/wan_video_camera_controller.py:8`、`DiffSynth-Studio/diffsynth/pipelines/wan_video.py:1560`。

### 7.11 Video-As-Prompt：提示视频分支与时间表示

VAP Unit 将提示视频编码为 latent，并拼接首帧 mask/图像 latent 条件形成 `vap_hidden_state`，另生成 `vap_clip_feature` 和分支文本 `context_vap`。只有共享输入中的 `end_image` 非空时才启用尾帧条件；使用的图像来自 `vap_video[-1]`，不是目标视频的 end_image 像素本身。

VAP 的 prompt 实际从 `inputs_posi["vap_prompt"]` 和 `inputs_nega["negative_vap_prompt"]` 读取，缺失时使用空字符串。普通训练模块仅将主 prompt 放在正向字典；用 `extra_inputs` 传入 `vap_prompt` 会进入共享字典，并不自动成为分支 prompt。现有 VAP 训练脚本不传这些 prompt 字段，因而走空文本默认路径。

前向按 `vap.mot_layers_mapping` 交错执行 VAP 与主干 block。时间表示有一个需要保留的具体区别：主干先用当前 timestep 生成 `t` 和 `t_mod`；进入 VAP 分支后，以常量 `clean_timestep=1` 生成 VAP 时间 embedding，并覆写同名变量 `t`，再得到 `t_mod_vap`。已有主干 `t_mod` 不被重算，而末端 `dit.head(x,t)` 接收到的是覆写后的 VAP `t`。这是当前代码的变量与数值流，不代表本文验证过它对生成质量的影响。

full 示例只训练 `vap` 并去掉 `pipe.vap.` 前缀；LoRA 示例训练 `dit` 上的 LoRA。二者的产物加载目标不同。

依据：`DiffSynth-Studio/diffsynth/pipelines/wan_video.py:738`、`DiffSynth-Studio/diffsynth/pipelines/wan_video.py:1628`、`DiffSynth-Studio/diffsynth/pipelines/wan_video.py:1714`、`DiffSynth-Studio/examples/wanvideo/model_training/full/Video-As-Prompt-Wan2.1-14B.sh:1`、`DiffSynth-Studio/examples/wanvideo/model_training/lora/Video-As-Prompt-Wan2.1-14B.sh:1`。

### 7.12 WanToDance：音乐特征与独立时间控制

WanToDance 不复用 S2V 的学习式语音 encoder。其音乐输入由 librosa 提取 onset envelope、20 维 MFCC、12 维 chroma、peak 和 beat 标记，共 35 维。全局音乐分支的构造参数为 `latent_dim=256`；前向中的固定插值尺寸 `(149,4800)` 不是原始特征维数或该 encoder 的 hidden 维数。随后特征继续插值到 `frame_num*8` 和文本 hidden 维度，并保存在 `dit.merged_audio_emb` 供 block 注入使用。

`wantodance_fps` 默认 30，在启用相应 dynamicfps/unimodel 分支时参与时间 RoPE/网络路径选择，独立于最终 `save_video` 的播放 FPS。音乐提取调用却没有传入这个参数：使用方法默认 30，以 `sr=fps*512` 加载音频，然后将局部 `sr` 改为 22050 后进行后续特征提取。这两套 FPS/采样率传递不能视为一个统一开关。

global 模型的负向输入还带 `skip_9th_layer=True`，前向跳过零起算编号为 9 的 block；这不只是更换负向文本。参考图和关键帧由后续独立 Unit 编码，关键帧还形成索引 mask，不能将音乐、参考图、关键帧视为一个通用音频条件字段。

依据：`DiffSynth-Studio/diffsynth/pipelines/wan_video.py:1075`、`DiffSynth-Studio/diffsynth/models/wan_video_dit.py:424`、`DiffSynth-Studio/diffsynth/pipelines/wan_video.py:1652`。

## 8. 实际网络前向

### 8.1 不等同于直接调用 dit.forward

常规主干的 `model_fn_wan_video` 直接使用 DiT 子模块：

```text
timestep → sinusoidal embedding → time_embedding → time_projection
context → text_embedding
latents + 可选 y → patchify
可选图像特征 → 图像投影 → context 拼接
视频 token + 时空 RoPE
  → blocks 循环
  → head
  → unpatchify
```

因此，`WanModel.forward` 的存在不意味着标准训练与推理一定通过整个该方法。决定这条链路的是 `pipe.model_fn`，默认值为 `model_fn_wan_video`。

### 8.2 DiTBlock

常规 block 主要包含：

- 带时空 RoPE 的 self-attention。
- 以文本及可选图像特征为条件的 cross-attention。
- FFN。
- 时间调制生成的 shift、scale、gate 与残差连接。

常规 LoRA 目标名 `q,k,v,o,ffn.0,ffn.2` 对应网络中的投影和 FFN 层。模块是否匹配及实际可训练参数集合仍取决于具体模型结构和 PEFT 注入结果。

### 8.3 分派分支

`model_fn_wan_video` 不只处理普通 WanModel：

- 同时提供 sliding window size 和 stride 时，先进入时间滑窗封装。
- `LongCatVideoTransformer3DModel` 进入 LongCat 前向。
- `WanAnimate2Transformer` 进入 Animate2 前向。
- 存在 `audio_embeds` 时进入 S2V 前向。
- 其他情况走常规 Wan 主干及其可选附加模块。

函数的 `**kwargs` 允许额外键进入，但并不保证额外键参与计算。滑窗封装在当前源码中显式组装一部分 `model_kwargs`，也不是无条件转发全部输入。各功能是否真正组合生效，需要以对应分支的数据传递为准。

### 8.4 Attention 实现

常规 `WanModel` 的 self/cross attention 使用 `wan_video_dit.py` 内部的 `flash_attention`，默认探测顺序为 FlashAttention 3、FlashAttention 2、SageAttention、PyTorch SDPA；compatibility mode 直接使用 SDPA。

这条路径不是对仓库 `core/attention/attention.py` 中全局实现选择器的简单调用。专项模型和 USP 替换可能使用不同执行路径。安装某个 attention 包会影响选择，但本文没有对这些后端的训练梯度和性能作逐项验证。

## 9. 训练模块生命周期

`WanTrainingModule.__init__` 的主要顺序：

1. 检查 gradient checkpointing 参数；关闭时给出 warning 并强制打开。
2. 解析模型、tokenizer、audio processor 配置。
3. 创建 `WanVideoPipeline`，计算 dtype 为 BF16。
4. 根据 task 拆分 PipelineUnit。
5. 调用 `resume_from_checkpoint`。
6. 设置训练 scheduler，冻结/解冻模块，加载 preset LoRA，注入训练 LoRA，加载 LoRA checkpoint。
7. 保存 task、额外输入及 timestep boundary 等设置。

模型默认在 Accelerate 设备初始化；指定 `initialize_model_on_cpu` 或启用逐层 model CPU offload 时先在 CPU 初始化。

### 9.1 全量训练与 LoRA

`switch_pipe_to_training_mode` 先调用 `pipe.scheduler.set_timesteps(1000, training=True)`，再调用 `freeze_except`：

- 全 Pipeline 先进入 eval，并关闭参数梯度。
- `trainable_models` 指定的模块进入 train 并开启参数梯度。
- 指定 `lora_base_model` 时，在目标模型注入 PEFT LoRA，LoRA 参数进入优化器参数集合。
- `:data_process` 阶段不注入训练 LoRA。

CLI 默认 LoRA rank 为 32，目标模块为 `q,k,v,o,ffn.0,ffn.2`。底层训练模块也支持空目标字符串时自动寻找目标，但 CLI 默认已经给出了明确列表。

`trainable_models` 和 `lora_base_model` 是不同配置，并没有被入口强制设为互斥。最终优化哪些参数，以实际 `requires_grad` 状态为准。

### 9.2 task 分派

| task | Unit 使用方式 | 最终行为 |
|---|---|---|
| `sft` | 普通 Unit 列表 | FlowMatch SFT |
| `sft:data_process` | 可预处理子图 | 返回中间输入并保存缓存 |
| `sft:train` | 与可训练模型相关的子图 | 从缓存做 FlowMatch SFT |
| `direct_distill` | 普通 Unit 列表 | 直接蒸馏损失 |
| `direct_distill:data_process` | 可预处理子图 | 保存蒸馏输入缓存 |
| `direct_distill:train` | 训练相关子图 | 从缓存做直接蒸馏 |

这是 Wan 入口当前注册的任务集合。其他模型训练目录存在更多算法，不代表 Wan 入口已经注册同样的任务。

### 9.3 forward

普通输入先执行 `get_pipeline_inputs`；缓存输入直接使用传入的三字典。随后递归移动输入到 Pipeline 的设备和 dtype，逐个执行 `pipe.units`，最后通过 task 对应的 loss 计算结果。

训练没有经过 `WanVideoPipeline.__call__` 的 `torch.no_grad` 装饰器，因此常规训练不依赖该推理接口建立反向图。

### 9.4 CLI 注册、主程序读取与 Pipeline 输入是三层关系

`add_general_config` 注册共享训练参数，但 Wan 主程序没有读取其中的 `template_model_id_or_path` 和 `enable_lora_hot_loading`。这些参数能被 parser 接受，不表示当前 Wan 启动流程会执行训练模板或相应开关。

`--framewise_decoding` 则由主程序读取，用于改变 Dataset 的时间整除参数；它没有直接作为布尔值传给 `WanTrainingModule` 的共享输入。WanToDance global 示例同时在 `extra_inputs` 列出 `framewise_decoding`，由 metadata 把该值送入 Pipeline。

因此此处有两个不同控制点：CLI 控制数据采样时间约束，metadata/shared input 控制 VAE 的 framewise 路径和语义帧数。只看到命令行出现该 flag，不能推断两处均已启用。

依据：`DiffSynth-Studio/examples/wanvideo/model_training/train.py:118`、`DiffSynth-Studio/examples/wanvideo/model_training/lora/Wan-Dancer-14B-global.sh:1`。

## 10. Flow Matching 训练目标与 scheduler

### 10.1 调度数组

Wan 调度先建立从噪声起点到零的等间隔 sigma 数组，去掉终点零，再应用 shift：

```text
sigma_start = denoising_strength
sigma_raw = linspace(sigma_start, 0, num_steps + 1)[:-1]
sigma = shift * sigma_raw / (1 + (shift - 1) * sigma_raw)
timestep = 1000 * sigma
```

训练使用 1000 个点，默认 shift 为 5；推理接口默认 50 个点，`sigma_shift=5.0`。

训练随机采样的是数组索引，不是直接在变换后的连续 timestep 上均匀采样。因此索引比例不等于 timestep 比例。

### 10.2 普通 SFT

令干净视频 latent 为 `z`，随机噪声为 `epsilon`，当前 sigma 为 `sigma`：

```text
z_sigma = (1 - sigma) * z + sigma * epsilon
target = epsilon - z
prediction = model_fn(z_sigma, timestep, conditions)
loss = MSE(prediction.float(), target.float()) * training_weight(timestep)
```

源码变量名为 `noise_pred`，但监督目标是流速度 `epsilon - z`，并非纯噪声 `epsilon`。

`max_timestep_boundary`、`min_timestep_boundary` 先乘调度数组长度并转整数，作为 `torch.randint` 的索引区间。上界不包含在采样范围中。

TI2V 首帧固定路径在加噪后写回 `first_frame_latents`，并从 prediction 与 target 中排除第一帧后再计算 MSE。

### 10.3 损失权重

训练权重不是常数。scheduler 在训练初始化时按 timestep 构建以中部为中心的曲线，减去最小值并归一化。`training_weight` 按当前 timestep 找到对应权重。

因此，训练日志的 loss 已经包含时间步权重，不等同于未加权 latent MSE。

### 10.4 推理更新

```text
z_next = z_current + prediction * (sigma_next - sigma_current)
```

最后一步以 `sigma_next=0` 结束。这是该实现的单步更新规则，不能把其他 diffusion scheduler 的 prediction 参数化或 step API 直接当作同一含义。

### 10.5 直接蒸馏

`DirectDistillLoss` 根据输入步数重新设置 scheduler，保持训练状态，通过少步模型前向和更新得到最终 latent，再与 `input_latents` 计算 MSE。

该函数不在线运行 teacher；目标视频来自数据。它也没有复用完整推理接口中的 CFG 采样流程。额外输入中存在 `cfg_scale` 不意味着这段损失实现已经执行同样的正负向 guidance。

仓库文档 `DiffSynth-Studio/docs/zh/Training/Direct_Distill.md:1` 标注该方案停止维护。源码仍保留任务入口和示例。

### 10.6 scheduler.training 与模型 train/eval 状态独立

`scheduler.training` 决定 InputVideoEmbedder 是返回监督 latent 还是执行 Video-to-Video 加噪，Animate2 也用它选择是否预计算参考 KV。

它由 `scheduler.set_timesteps(..., training=...)` 管理，不由 `pipe.train()` 或 `pipe.eval()` 自动管理。训练初始化设置为 True；普通推理调用重设调度并置为 False；直接蒸馏先重设步数再显式置 True。

Pipeline 因而是包含可变 scheduler 状态的对象，不是只有权重的无状态函数。训练和推理复用同一实例时，前一次调用的采样网格及 training 标志不会自动恢复。共享实例并发调用的隔离性也不能由 no-grad 推断得到；当前实现没有围绕这些状态提供并发锁或每请求独立 scheduler。

### 10.7 BF16 timestep 与实际加噪网格

标准 SFT 先按索引从 FP32 scheduler 数组取 timestep，再转成 `pipe.torch_dtype`，当前训练入口为 BF16。随后 `add_noise` 和 `training_weight` 按这个已转换 timestep 在原调度数组中寻找最近位置，而不是直接沿用最初抽到的索引。

这意味着连续数学网格、抽样索引和实际被查询的噪声网格需要区分。例如 1000 点、shift=5 的当前 CPU scheduler 实测：

| 抽样索引 | 原始 timestep | 转成 BF16 后 | 最近网格索引 | 该网格 timestep |
|---:|---:|---:|---:|---:|
| 416 | 约 875.300 | 876 | 414 | 约 876.196 |
| 417 | 约 874.850 | 876 | 414 | 约 876.196 |
| 357 | 约 900.056 | 900 | 357 | 约 900.056 |
| 358 | 约 899.664 | 900 | 357 | 约 900.056 |

两侧相邻索引可以在 BF16 转换后落到相同位置。因此 A14B 的配置表精确描述的是抽样索引分区，不应进一步解释成实际加噪 timestep 在边界处严格互斥。该数值行为有 scheduler 隔离检查；其训练质量影响未作模型级实验。

推理专家切换在循环中先比较原 scheduler timestep，再将其转换成模型 dtype，不能把训练中的最近网格查询过程直接套到推理切换判断上。

依据：`DiffSynth-Studio/diffsynth/diffusion/loss.py:5`、`DiffSynth-Studio/diffsynth/diffusion/flow_match.py:457`、`DiffSynth-Studio/diffsynth/pipelines/wan_video.py:330`。

## 11. Wan2.2 A14B 的双模型机制

### 11.1 训练和推理中的属性含义不同

训练脚本分两次启动：第一次只加载 high-noise DiT，第二次只加载 low-noise DiT。每次只有一个 DiT，所以两次训练都使用 `pipe.dit`。

推理同时加载两个 DiT：先加载 high-noise 到 `pipe.dit`，再加载 low-noise 到 `pipe.dit2`。这不是一次前向中同时组合两个专家，也不是训练循环内自动切换专家。

### 11.2 配置对应关系

| 模型 | high-noise 索引比例 | low-noise 索引比例 | 推理阈值 |
|---|---|---|---|
| Wan2.2 T2V-A14B | `[0, 0.417)` | `[0.417, 1)` | `switch_DiT_boundary=0.875` |
| Wan2.2 I2V-A14B | `[0, 0.358)` | `[0.358, 1)` | 普通推理示例显式设为 `0.9` |

训练配置来源：`DiffSynth-Studio/examples/wanvideo/model_training/lora/Wan2.2-T2V-A14B.sh:1`、`DiffSynth-Studio/examples/wanvideo/model_training/lora/Wan2.2-I2V-A14B.sh:1`。

索引与 timestep 的数值对应为：

| 数组索引 | shift=5 的 timestep |
|---:|---:|
| 416 | 约 875.300 |
| 417 | 约 874.850 |
| 357 | 约 900.056 |
| 358 | 约 899.664 |

`0.417`、`0.358` 是数组位置比例；`0.875`、`0.9` 是归一化 timestep 阈值。改变训练 scheduler 的 shift 或采样网格时，这组对应关系不再自动成立。

表中的 timestep 是未转模型 dtype 的调度网格；标准 SFT 的 BF16 转换和最近网格查找还会改变实际查询位置，具体见第 10.7 节。

### 11.3 推理切换

推理循环先使用 `in_iteration_models`。当 `timestep < switch_DiT_boundary * 1000` 且存在 `dit2`，则把当前 `models['dit']` 切换为 `dit2`，同时将 VACE 模型槽位切到 `vace2`。

循环单向从高噪声向低噪声推进，不反复切回高噪声模型。Video-to-Video 若从较低噪声起点开始，可以在循环初期就满足切换条件。

### 11.4 两份产物的加载

因为两次训练都保存 `pipe.dit`，两个 checkpoint 都可以去掉 `pipe.dit.` 前缀。推理端通过加载目标区分它们：

```python
pipe.load_lora(pipe.dit, high_noise_checkpoint, alpha=1)
pipe.load_lora(pipe.dit2, low_noise_checkpoint, alpha=1)
```

全量权重对应加载到 `pipe.dit` 与 `pipe.dit2`。交换两个文件或交换模型配置顺序，不会被模型槽位名称自动纠正。

## 12. 优化循环、步数与日志

### 12.1 当前公共训练循环

```text
创建 optimizer(trainable_modules)
创建 ConstantLR(optimizer)
创建 shuffle=True 的 DataLoader
Accelerate prepare 或逐层 offload 初始化
for epoch:
  for batch:
    accelerator.accumulate(model)
      forward
      accelerator.backward(loss)
      可选 offload after_backward
      optimizer.step
      scheduler.step
      optimizer.zero_grad
      ModelLogger.on_step_end
  按配置保存 epoch checkpoint
训练结束保存未落盘的最后一个 step checkpoint
```

默认 optimizer 为 AdamW，可通过 `customized_optimizer` 指定可导入的优化器类。公共循环没有显式的梯度裁剪、EMA、自动验证调用或 cosine 学习率逻辑。

学习率 scheduler 是 `torch.optim.lr_scheduler.ConstantLR(optimizer)`，没有显式提供 factor 或 total_iters。它使用实际 PyTorch 版本的默认参数，不能仅凭类名解释成学习率从第一步起始终固定。

### 12.2 Batch 与梯度累积

DataLoader 未传 `batch_size`，当前每进程每次加载一个样本；`collate_fn=lambda x: x[0]` 直接返回第一个样本。

在完整且均匀的累积窗口内，有效 batch 可按 `进程数 × gradient_accumulation_steps` 理解。尾部窗口及分布式采样补齐行为由 Accelerate 管理。

直接增大 DataLoader 的 batch size，而保留当前 collate 函数，并不会形成有效多样本 batch。

### 12.3 Step 的含义

`ModelLogger.num_steps` 在每次 batch 调用 `on_step_end` 时加一，所以它是微批次计数，不是独立记录的优化器更新计数。

- `save_steps` 按这个计数触发。
- loss 日志也使用这个计数。
- 梯度累积后，同一个“step”命名不能直接视为一次有效参数更新。

日志只在主进程初始化和写入。传入的 loss 是当前进程的 loss；当前 logger 没有显式跨进程求平均。

### 12.4 保存文件

- 未设置 `save_steps`：每个 epoch 结束保存 `epoch-{epoch_id}.safetensors`，epoch 从 0 开始。
- 设置 `save_steps`：周期保存 `step-{num_steps}.safetensors`，结束时补存尚未落盘的最后一步。
- 正常训练启动时保存 `training_args.json`。
- 按配置输出 TensorBoard、SwanLab、W&B 或 `loss.csv`。

示例训练 5 个 epoch 时，最终 epoch 文件名为 `epoch-4.safetensors`；2 个 epoch 时为 `epoch-1.safetensors`。

## 13. Split training 与缓存

### 13.1 拆分依据

`PipelineUnitGraph` 根据 Unit 声明的输入、输出和 `onload_model_names` 建立依赖关系。它从训练相关模型对应的 Unit 出发，沿数据依赖和变量更新链扩展训练相关子图。

- `:data_process` 执行其他可预处理 Unit。
- `:train` 执行训练相关 Unit。
- 可训练模型名称来自 `trainable_models` 和 `lora_base_model`。

这是声明式计算图拆分，不是根据某次运行自动分析 tensor 是否 requires_grad 的通用图追踪。

### 13.2 缓存读写

缓存阶段使用 no-grad，保存训练模块返回的三个输入字典：

```text
output_path/
  0/0.pth
  0/1.pth
  ...
  其他进程编号/样本编号.pth
```

缓存训练省略 metadata 路径。`UnifiedDataset` 递归查找 `dataset_base_path` 中的 `.pth` 文件并读取。省略 metadata 的含义是切换到缓存模式，不是自动从目录扫描原始视频。

当前 Wan 拆图默认未开启 `remove_unnecessary_params`，缓存可以保留较多共享输入，而不只是文本 embedding 与视频 latent。原始输入、运行参数或不再需要的中间状态也可能出现在缓存中。

`offload_models` 配置改变模型加载和参数保留方式，不等于完全不构造该模型或不读取其结构。

### 13.3 噪声与配置

标准 SFT loss 每次重新调用 `randn_like(input_latents)`。缓存里即使有 NoiseInitializer 的噪声，也不意味着普通 SFT 固定使用缓存噪声。

另一方面，缓存路径直接使用已保存字典，不重新调用 `get_pipeline_inputs`。因此缓存中的 timestep boundary、CFG、条件配置等可以继续生效，而不会自动被新命令行配置重建。

编码器权重、tokenizer、尺寸/FPS/片段选择或条件语义变化后，已有缓存也不会自动失效。当前实现没有缓存 fingerprint 或自动一致性校验。

缓存使用 `torch.load(..., weights_only=False)` 读取，以支持包括非张量对象在内的输入字典；缓存文件应来自可信来源。

缓存文件名由当前 `process_index` 和 DataLoader 的枚举编号构成，不是稳定的原始样本 ID。再次向同一输出目录处理数据时，同名文件被写入，但没有清理旧目录中剩余 `.pth`；下一次缓存扫描会读取仍存在的所有文件。数据量、分布式进程数或样本排列变化后，目录内容不能仅凭文件编号认定属于同一次预处理结果。

依据：`DiffSynth-Studio/diffsynth/core/data/unified_dataset.py:63`、`DiffSynth-Studio/diffsynth/diffusion/runner.py:119`。

示例：`DiffSynth-Studio/examples/wanvideo/model_training/special/split_training/Wan2.1-T2V-1.3B.sh:1`。

## 14. 推理过程与默认值

### 14.1 默认参数

| 参数 | 当前默认值 |
|---|---|
| `height`、`width` | 480、832 |
| `num_frames` | 81 |
| `num_inference_steps` | 50 |
| `cfg_scale` | 5.0 |
| `cfg_merge` | False |
| `sigma_shift` | 5.0 |
| `switch_DiT_boundary` | 0.875 |
| `denoising_strength` | 1.0 |
| `rand_device` | cpu |
| `seed` | None |
| `tiled` | True |
| `tile_size`、`tile_stride` | `(30, 52)`、`(15, 26)` |
| `output_type` | `quantized` |

这些是接口默认值，不是所有模型的最优参数。专项示例可以覆盖它们。训练共享输入默认 `tiled=False`、`cfg_scale=1`、随机设备为 Pipeline 设备，与推理默认值不同。

### 14.2 采样流程

1. 按步数、denoising strength 和 shift 设置 scheduler。
2. 组装三个输入字典，执行 Unit 列表。
3. 加载迭代模型，开始 timestep 循环。
4. 需要时切换高/低噪声模型。
5. 计算正向预测；CFG 不为 1 时计算或拆出负向预测。
6. 合成预测并调用 scheduler 更新 `latents`。
7. 需要时重新写入固定首帧。
8. 处理专项参考帧并执行 post units。
9. VAE 解码。
10. 转成视频帧或保留浮点张量，结束时调用卸载接口。

整个推理接口有 `torch.no_grad` 装饰。

### 14.3 CFG

```text
prediction = prediction_negative
             + cfg_scale * (prediction_positive - prediction_negative)
```

`cfg_merge=False` 时分别运行正、负分支。启用 merge 时合并条件 batch，一次前向后在 batch 维拆成两份预测。

merge 不是“免除负向计算”，而是改变执行组织和 batch 大小。实际显存与速度取决于模型及后端。

合并 Unit 仅处理 `context`、`clip_feature`、`y`、`reference_latents` 四个名字：有正负 tensor 时拼接，否则若共享 tensor 存在则复制共享值。之后直接清空正、负字典，不通用地保留其他字段。此前生成的分支 `audio_embeds`、`face_pixel_values`、`context_vap`、`tea_cache` 及分支标志等不会自动合并；共享字典的其他字段则不会被这个 clear 删除。

顺序也有影响：WanToDance 等后续 Unit 可以重新写入分支字段，但 merge 推理不会独立执行负向 model_fn，所以不能据此认为其负向 `skip_9th_layer` 等行为得到保留。默认关闭 merge；专项条件组合的语义必须按字段的实际来源判断，不能仅依据模型支持普通 CFG。这里验证的是字段流，不是所有专项模型组合的完整生成结果。

依据：`DiffSynth-Studio/diffsynth/pipelines/wan_video.py:844`、`DiffSynth-Studio/diffsynth/pipelines/wan_video.py:338`。

### 14.4 输出

`quantized` 输出通过 `vae_output_to_video` 转为视频帧。`floatpoint` 保留解码浮点输出；该参数名在源码中就是 `floatpoint`。

`save_video` 逐帧写出 MP4。示例常用 `fps=15`、`quality=5`；保存 FPS 是文件播放参数，不证明训练输入按该 FPS 采样，也不代表模型显式接受了相同 FPS 条件。

默认 `vae_output_to_video` 将 `[B, C, T, H, W]` 转为 `[T, H, W, C]` 时，对没有出现在目标 pattern 中的维度执行 mean reduction。当前 B=1 路径符合预期，但 B>1 时默认行为是沿 batch 求平均，不是返回多个独立视频。常规外部输入、内部 batch 和输出转换都不能仅通过扩大某一维度当作完整批量生成支持。

固定 seed 只约束指定路径的随机数。随机设备、模型参数、采样设置、条件、attention 后端及近似缓存等变化，均可能改变输出；本文未做跨设备或跨后端的 bitwise 一致性验证。

## 15. 权重保存、加载与恢复

### 15.1 保存的是可训练参数

`ModelLogger.save_model` 从 Accelerate 获取 state dict，再经 `export_trainable_state_dict` 按可训练参数名筛选。

因此：

- LoRA 训练产物通常只包含 LoRA 可训练参数。
- `trainable_models=dit` 保存 DiT 的可训练权重，不包含整套冻结文本编码器和 VAE。
- 新增或附加模型只有实际进入可训练参数集合，才会被该导出逻辑包含。
- 产物不是自动可独立加载的完整 Pipeline 包。

### 15.2 前缀

训练模块持有 `pipe`，参数全名例如 `pipe.dit.blocks...`。`remove_prefix_in_ckpt='pipe.dit.'` 去除对应前缀，供推理端直接加载到 `pipe.dit`。

VACE 示例则使用 `pipe.vace.`。去前缀只对匹配的键生效，不会将一个多模块产物自动拆成多个文件。

LoRA 的 PEFT `.default` 等键格式由 LoRA 转换器处理。不能把 LoRA 字典直接当作完整 DiT state dict，也不能认为任意外部 LoRA 键名都无须转换。

### 15.3 推理加载

| 产物 | 当前示例使用方式 |
|---|---|
| LoRA | 构建基础 Pipeline，再 `pipe.load_lora(目标模块, 文件, alpha=...)` |
| 全量 DiT | 构建基础 Pipeline，再对目标 DiT `load_state_dict` |
| 双模型产物 | 分别加载到 `dit` 与 `dit2` |

`load_lora` 的 `alpha` 是应用强度。训练端 LoRA rank 和 alpha 默认相等；外部 LoRA 文件自带 alpha 时，转换器还有相应格式兼容逻辑。这几个值不是一个含义。

`preset_lora_path` 与 `lora_checkpoint` 也不同：前者在注入新的训练 LoRA 前通过 Pipeline 加载，影响基础模型上的预设适配；后者加载到新注入的训练 LoRA 参数中。它们不会自动形成一个统一“续训 checkpoint”。

### 15.4 融合与 hotload

没有显存管理时，`load_lora` 默认融合到基础权重。启用相应显存包装时，默认可以走 hotload，将 LoRA 权重存入包装层的列表，在计算时应用。

- 融合会实际改变基础权重。
- 重复融合是叠加，不是自动替换上一次 LoRA。
- `clear_lora` 只清除 hotload 列表，不恢复已经融合的基础权重。
- hotload 也可以追加多个 LoRA，不能把反复调用视为自动只保留最后一个。

这一点由 `BasePipeline.load_lora`、`clear_lora` 以及 `GeneralLoRALoader.fuse_lora_to_base_model` 的实现决定。

### 15.5 恢复训练

`resume_from_checkpoint` 加载权重、按配置加回前缀，并以 `strict=False` 加载到训练模块；unexpected keys 会触发异常，missing keys 不在同一方式下被拒绝。

它在 LoRA 注入前执行。`lora_checkpoint` 则在 LoRA 注入后做键转换并加载到目标模型。这是两个不同阶段、不同用途的加载接口。

现有 checkpoint 不保存 optimizer、学习率 scheduler、随机状态、epoch 或步数状态。`resume_from_checkpoint` 因此是权重恢复，不是完整的运行状态续训。

### 15.6 验证示例

`validate_lora` 和 `validate_full` 是独立生成脚本，不由训练循环自动调用。示例文件路径和 epoch 名称是写死的，且不自动读取 `training_args.json` 重建验证配置。

它们用于加载训练产物后生成视频，不构成训练集/验证集管理、自动指标计算或模型选择系统。

## 16. 显存管理、训练 offload 与分布式

### 16.1 三种不同机制

| 机制 | 控制对象 | 当前接入位置 |
|---|---|---|
| Pipeline 显存管理 | 模型/包装层的存储与计算位置、dtype | ModelConfig、VRAM wrapper、Unit/model 阶段切换 |
| 逐层训练 offload | 训练前向/反向期间的参数、buffer 和可选梯度/优化器位置 | `OffloadTrainingManager` |
| DeepSpeed | 分布式训练参数/梯度/优化器状态管理 | Accelerate 配置与 prepare |

同样出现 CPU offload 字样，不代表它们是同一实现，也不能直接推断这些配置任意叠加后都兼容。

### 16.2 Pipeline 显存管理

ModelConfig 可以分别设置 offload、onload、preparing、computation 的 dtype 和 device。低显存示例会明确给出这些配置，例如磁盘存储、CPU 装载、GPU 准备和 BF16 GPU 计算。

模型包装映射决定哪些模块被替换成自动装载/卸载层。WanModel 的映射包含 Linear、Conv、Norm、DiTBlock、Head 等。

`load_models_to_device` 只在显存管理已启用时执行相应装载/卸载。因此，普通配置中出现该调用，并不意味着未包装的全部模型会自动在 CPU/GPU 间移动。

`vram_limit` 是显存管理使用的配置，不是单独提供该值就自动把任何模型变成分层 offload 模型；ModelPool 还检查 offload dtype 和 device 等设置。

### 16.3 逐层训练 offload

启用 `enable_model_cpu_offload` 后，runner 不通过 `accelerator.prepare(model)` 包装模型，而是准备 optimizer、DataLoader、scheduler，再初始化 OffloadTrainingManager。

manager 使用 hook 在前向、反向和重计算阶段管理权重。被冻结参数与可训练参数采用不同策略；是否启用 optimizer CPU offload 也改变可训练参数和梯度的位置。

本 commit 的文档 `DiffSynth-Studio/docs/zh/Training/Offload_Training.md:1` 标注该机制当前只支持单卡，不兼容 DDP 场景。

### 16.4 Gradient checkpointing

常规 DiT block 通过 `gradient_checkpoint_forward` 包装。普通 PyTorch checkpoint 路径使用 `use_reentrant=False`；启用 checkpointing offload 时结合 `save_on_cpu`。DeepSpeed activation checkpointing 已配置时存在另一条分支。

checkpointing 降低激活保存成本，不等于减少基础模型权重。它会改变前向重计算行为。

### 16.5 DeepSpeed 示例

`DiffSynth-Studio/examples/wanvideo/model_training/full/accelerate_config_14B.yaml:1` 配置 8 进程、BF16、ZeRO Stage 2，并声明 CPU offload 相关项。

该文件不是 ZeRO Stage 3 配置。参数、梯度和优化器实际如何分片或 offload，必须结合所选 ZeRO stage 及 DeepSpeed 的有效配置理解，不能仅根据 YAML 中存在某个字段推断其实际生效范围。

### 16.6 USP、TeaCache 与 VAE tiling

- USP 是推理序列并行：拆分 token 计算，替换部分 attention/forward 路径，并在需要时 gather。示例用 torchrun 启动，多卡共享一次生成任务。
- DDP 是训练数据并行，不等于 USP。
- TeaCache 在满足条件时复用部分计算，是可选的推理近似路径；默认不启用。
- VAE tiling 在编码/解码阶段分块并融合，不拆分 DiT 的全视频 token self-attention。
- Temporal sliding window 是模型前向的时间滑窗封装，与 VAE 空间 tiling 不同。

低显存和加速示例提供特定配置，不构成所有专项模型、量化、LoRA、USP 和缓存组合的兼容性保证。

### 16.7 FP8、量化模式及明确的组合限制

训练的 `fp8_models` 通过显存包装配置设置 FP8 存储/准备 dtype、BF16 计算 dtype。它不是直接把所有输入、激活或可训练 LoRA 都改成 FP8。

`parse_vram_config` 先判断 FP8，再判断 offload。因此同一模型同时匹配这两个列表时，返回 FP8 配置，不进入带 `clear_parameters=True` 的磁盘 offload 分支。Split training 示例的清参数配置也不能泛化成“任意训练时把可训练模型清空”。

`quant_options` 使用另一套 QuantizeConfig/后端体系：

| 配置或组合 | 当前实现 |
|---|---|
| `mode='dynamic'` | 保留后端量化 Linear，默认模式 |
| `mode='dequant_once'` | 量化后恢复成普通浮点 Linear，保留量化误差；不再等同于持续量化权重计算 |
| `load_prequantized=True` | 先建立量化权重结构，再加载预量化 checkpoint 及必要 metadata |
| 磁盘 offload + 在线量化 | loader 显式拒绝；该组合要求预量化 checkpoint |
| backend 声明不可微 | 训练配置解析显式拒绝，包括冻结基础层上的 LoRA 训练 |
| 注册为预量化模型的文件 + 用户量化参数 | ModelPool 使用注册量化配置，打印提示并忽略传入的量化配置 |

冻结量化层仍可能需要将梯度传向 LoRA 或其他可训练模块，所以“不更新基础权重”不意味着量化前向无需具备梯度传播能力。

上述可微性检查位于显式 `quant_options` 的解析流程，不是对最终加载模型做一次实际梯度验证。模型注册表自动选择的预量化配置走 ModelPool 的另一条配置来源，不能把这一入口检查理解成所有量化模型已经通过训练兼容性测试。

量化模型调用 `load_lora` 时，`check_quant_hot_load` 会确保相应包装支持 hotload，因为量化权重不能像普通浮点权重那样直接吸收 LoRA 融合。普通 `lora_checkpoint` 的 PEFT 训练加载与这一推理 hotload 路径仍是不同机制。

DDP 初始化前还会尝试排除被冻结、由可追踪 tensor subclass 承载的量化权重，避免广播桶处理这些权重；这不是将全部冻结参数都从 DDP 同步中统一删除。

依据：`DiffSynth-Studio/diffsynth/diffusion/training_module.py:153`、`DiffSynth-Studio/diffsynth/diffusion/training_module.py:180`、`DiffSynth-Studio/diffsynth/core/quant/config.py:88`、`DiffSynth-Studio/diffsynth/core/loader/model.py:17`、`DiffSynth-Studio/diffsynth/models/model_loader.py:71`、`DiffSynth-Studio/diffsynth/diffusion/base_pipeline.py:243`。

### 16.8 区域编译而非只编译顶层 forward

Pipeline 提供 `compile_pipeline`；Wan 声明 `compilable_models=['dit', 'dit2']`。声明可编译模型不表示普通加载过程已经自动调用编译。

对于具有 `_repeated_blocks` 的模型，编译器遍历模块，按类名匹配重复 block 并调用各 block 的 compile。普通 WanModel 声明 `_repeated_blocks=['DiTBlock']`，因此使用区域编译；没有该属性的模型才走整个模型的 compile。

这与常规 `model_fn_wan_video` 手动遍历 DiT blocks 相吻合。顶层 `dit.forward` 不是主执行入口，不意味着仓库的区域编译无法覆盖实际 block 计算。具体编译性能与 graph break 情况未做模型级实测。

依据：`DiffSynth-Studio/diffsynth/diffusion/base_pipeline.py:384`、`DiffSynth-Studio/diffsynth/models/wan_video_dit.py:338`。

### 16.9 TeaCache 复用的是 block 残差

启用条件是 `tea_cache_l1_thresh` 非空，且必须显式选择受支持的 coefficient ID；它不根据实际加载权重自动识别。当前只接受：

```text
Wan2.1-T2V-1.3B
Wan2.1-T2V-14B
Wan2.1-I2V-14B-480P
Wan2.1-I2V-14B-720P
```

空 ID 或其他名称会在构造时抛出 ValueError。Unit 每次预处理生成新对象，普通 CFG 下正负分支独立；`cfg_merge` 会清空这些分支缓存对象。

每条缓存的首步和末步强制计算中间 blocks。其他步对 `t_mod` 相对变化应用系数多项式并累积，与阈值比较决定计算或复用。缓存保存的是 `block_output - pre_block_hidden_states`；复用时加到当前 pre-block 特征上，不是直接重用上一轮完整 `noise_pred`。前置 embedding、patchify 和最终 head 仍执行。

VACE hints 在复用判断之后、实际跳过 block 的条件分支之前计算，因此缓存复用不免除该部分 VACE 前向。实际 block 内的 VACE 注入、Animate hook、WanToDance hook 等随 block 循环一同跳过。A14B 专家切换循环没有显式清空 TeaCache，而系数表也没有 Wan2.2 标签；不能据此认定旧系数与跨专家残差具有已验证的适用性。

依据：`DiffSynth-Studio/diffsynth/pipelines/wan_video.py:828`、`DiffSynth-Studio/diffsynth/pipelines/wan_video.py:1334`、`DiffSynth-Studio/diffsynth/pipelines/wan_video.py:1648`、`DiffSynth-Studio/diffsynth/pipelines/wan_video.py:1714`。

### 16.10 DiT 时间滑窗的传参范围

`sliding_window_size`、`sliding_window_stride` 同时非空才进入 TemporalTiler。两者以 latent 时间位置计数，不是原视频帧数。它按窗计算预测，以重叠区权重融合，并在正常结束时恢复传入字典中的原始 tensor。

当前 wrapper 只沿时间切片 `latents` 和 `y`。显式转发的其余参数是 `dit`、`motion_controller`、`vace`、`timestep`、`context`、`clip_feature`、`reference_latents`、`vace_context`、`vace_scale`、`tea_cache`、USP 标志和 `motion_bucket_id`；`vace_context` 等条件即使有时间维，也仍保持原长度。

音频、姿态、面部、animate adapter、VAP、音乐、相机特征、gradient checkpoint 开关等没有被该调用字典转发。外层 `**kwargs` 能接收它们，不代表滑窗递归会保留。每窗普通前向从 `dit.freqs[0][:f]` 构造时间 RoPE，没有转发全局窗口起始位置；TeaCache 对象若存在则在各窗间传递，计数随每次窗前向推进，并非独立的每个采样步计数。

因此这是一个有明确传参范围的前向封装，而非所有 Wan 专项条件都可直接启用的通用长视频模式。它与 VAE 空间 tiling、S2V 外部音频分段，以及 Animate2 长视频状态管理是不同机制。

依据：`DiffSynth-Studio/diffsynth/pipelines/wan_video.py:1387`、`DiffSynth-Studio/diffsynth/pipelines/wan_video.py:1496`。

## 17. 当前代码中的配置差异与边界

本节记录源码中直接可见的差异，以及由这些差异带来的使用约束。视频质量、性能和多配置组合的影响未在本文中通过端到端实验量化。

| 位置或行为 | 源码事实 | 使用含义 |
|---|---|---|
| I2V-A14B 验证阈值 | 普通 I2V 推理示例设 `0.9`；LoRA/full 验证脚本未设，使用默认 `0.875` | 它们的专家切换配置并不一致，不能当作相同采样设置 |
| I2V-A14B 验证图像 | 从 `data/example_video_dataset/video1.mp4` 取首帧 | 不是训练脚本下载目录的自动映射，文件是否存在需独立满足 |
| T2V-A14B full 验证长度 | 训练脚本指定 49 帧，full 验证未指定，使用默认 81 帧 | 验证没有自动复用训练长度 |
| 固定宽高 | Dataset 固定尺寸不自动整除对齐，Pipeline 可能向上调整字典尺寸 | 不满足实际 VAE/patch 约束时，尺寸声明与已加载视频可能不一致 |
| TI2V 动态尺寸 | 数据算子整除因子为 16，Pipeline 要求 32 | 某些动态分辨率不能直接满足两层约束 |
| 缓存训练输入 | 不重建 `get_pipeline_inputs` | 修改命令行参数不能自动重算缓存中的条件和 boundary |
| 恢复顺序 | 通用恢复在 LoRA 注入前；LoRA 恢复在注入后 | 两种 checkpoint 接口不能仅按名称视为等价 |
| 加载严格性 | 通用恢复允许 missing keys，但拒绝 unexpected keys | 加载成功不证明全部目标参数都被恢复 |
| LoRA 融合 | 加到基础权重；clear 只清 hotload 列表 | 同一进程中的反复加载会影响权重状态 |
| Batch collate | 只返回样本列表第一项 | 单独调大 batch size 不会实现多样本训练 |
| Step 命名 | Logger 按微批次计数 | 累积训练时 checkpoint/log step 不等于有效更新次数 |
| 输出 FPS | save_video 独立设置 | 输出播放速率不证明训练时统一 FPS |
| 额外输入 | model_fn 存在 `**kwargs`，专项分支选择性使用参数 | 参数能传入不证明它参与网络计算 |
| CFG merge | 只合并四组 tensor，随后清空分支字典 | 专项条件和缓存不会通用地保留 |
| 时间滑窗 | 仅切片 latents/y，并选择性转发条件 | latent 窗口不是通用的像素帧截取或全条件长视频推理 |
| Unit 拆图 | 名称精确匹配，依赖人工声明 | 子模块训练路径和运行时隐式字段不一定被图完整识别 |
| Animate full | 示例训练 animate_adapter | 不能按 full 目录名推断产物是完整 DiT |
| VAP 文本 | 从分支字典读取，不从共享字典读取 | extra_inputs 同名值不自动成为 VAP 的文本条件 |
| metadata 路径列表 | 默认视频算子只路由字符串 | JSON 列表与 Pipeline PIL 帧列表输入不等价 |
| 数据 schema | 部分文件键不存在时 Dataset 跳过，训练仍按必需键访问 | 入口没有完整的自动 schema 校验 |
| 缓存安全 | 使用非 weights-only pickle 加载 | 不应读取不可信来源的训练缓存 |
| 共享训练参数 | parser 注册 template/hot-loading 参数，但 Wan main 未读取 | CLI 被接受不代表行为被实现 |
| framewise | CLI 改采样约束，metadata/shared input 改编码路径 | 两层开关不等价 |
| 缓存目录复用 | 不删除前次剩余 `.pth` | 扫描结果可能包含不同次预处理的文件 |
| S2V 单次调用 | 普通音频 Unit 使用第一段 embedding | 覆盖长音频需要外部多段循环 |
| VACE / Animate2 参考槽 | 目标与条件增加参考槽，推理输出删除 | 与首帧从 loss 排除不是同一机制 |
| 默认视频转换 | batch 维被 mean reduction | 不自动返回 B 个视频 |
| 量化与磁盘 offload | 在线量化组合在 loader 被拒绝 | 需要符合预量化加载路径 |
| Pipeline 状态 | scheduler 被训练/推理调用改写 | train/eval 切换不自动恢复采样状态 |
| BF16 timestep | 加噪和权重按转换后的值重新找最近网格 | 专家索引分区不等于实际噪声等级严格互斥 |
| 多样本 tiled VAE encode | tile 参数在样本循环中累乘空间压缩倍数 | 不能认定同批各样本使用相同 tile 配置 |
| 音轨合成 | 单独调用 PATH 中的 ffmpeg，执行异常会被捕获 | 写出视频不证明合成音轨成功 |

对应验证源码：

- `DiffSynth-Studio/examples/wanvideo/model_inference/Wan2.2-I2V-A14B.py:1`
- `DiffSynth-Studio/examples/wanvideo/model_training/validate_lora/Wan2.2-I2V-A14B.py:21`
- `DiffSynth-Studio/examples/wanvideo/model_training/validate_full/Wan2.2-I2V-A14B.py:24`
- `DiffSynth-Studio/examples/wanvideo/model_training/validate_full/Wan2.2-T2V-A14B.py:1`

## 18. 现有运行入口

以下命令指向本 commit 已有脚本，不代表在任意硬件上均可直接运行。训练脚本包含数据下载、模型下载和较长训练配置；路径、资源和依赖需与脚本对应。

```bash
cd DiffSynth-Studio
python examples/wanvideo/model_inference/Wan2.1-T2V-1.3B.py
python examples/wanvideo/model_inference/Wan2.2-T2V-A14B.py
python examples/wanvideo/model_inference_low_vram/Wan2.2-T2V-A14B.py
```

```bash
bash examples/wanvideo/model_training/lora/Wan2.1-T2V-1.3B.sh
bash examples/wanvideo/model_training/lora/Wan2.2-T2V-A14B.sh
bash examples/wanvideo/model_training/full/Wan2.2-T2V-A14B.sh
```

```bash
python examples/wanvideo/model_training/validate_lora/Wan2.2-T2V-A14B.py
python examples/wanvideo/model_training/validate_full/Wan2.2-T2V-A14B.py
bash examples/wanvideo/model_training/special/split_training/Wan2.1-T2V-1.3B.sh
```

依赖以 `DiffSynth-Studio/pyproject.toml:1` 为准。基础依赖包含 torch、torchvision、transformers、imageio/ffmpeg、safetensors、modelscope、accelerate、peft 等；DeepSpeed、音频、量化、日志和 NPU 有各自 extras。extras 之间存在版本约束和冲突，不能把安装全部 extras 视为通用运行前提。

## 19. 验证范围

本次源码分析附带执行了以下轻量检查：

| 项目 | 结果 |
|---|---|
| 训练入口、Pipeline、公共训练模块、runner、logger、loss、scheduler、Dataset、数据算子共 9 文件的 AST 语法解析 | 通过 |
| 真实 Wan scheduler 的加噪公式与实现一致性 | 通过 |
| 训练 target 为 `epsilon - z` | 通过 |
| 理想 target 下 Euler 更新到零 sigma 重建干净 latent | 通过 |
| 抽查训练权重取值 | 通过 |
| `0.417` 索引切分对应 875 阈值两侧 | 通过 |
| `0.358` 索引切分对应 900 阈值两侧 | 通过 |

另外对源码提取的方法进行了 CPU 隔离检查；涉及网络计算的检查使用明确的替身对象，不加载完整模型：

| 项目 | 检查范围与结果 |
|---|---|
| 递归输入 dtype 转换 | FP32 转 BF16，FP64/整数保持；通过 |
| FP8/offload 配置优先级 | 同时启用时选择 FP8 分支；通过 |
| framewise 额外输入 | 不注入 metadata flag 时不重写语义帧数，注入后按公式重写；通过 |
| 缓存扫描 | 改写已有文件后，前次剩余文件仍被扫描；通过 |
| 区域编译分派 | 替身 DiTBlock 收到 compile，顶层模型不收到；通过，不涉及实际编译后执行 |
| S2V 音频分段选择 | 替身编码器验证默认第一段、16 FPS、80 帧分段及预计算绕过；通过 |
| VAE posterior mean 归一化 | 替身 encoder 验证首帧/四帧组织与按通道归一化公式；通过，不验证学习到的 VAE 数值 |
| 共享 CLI 参数读取 | AST 检查 Wan main 未消费 template/hot-loading 字段；通过 |
| BF16 timestep 最近网格 | 真实 scheduler 验证转换后与原抽样索引不同，边界相邻索引可合并；通过 |
| 多样本 VAE tile 参数 | 替身 tiled encoder 验证包装层跨样本累乘，未运行完整 VAE；通过 |
| CFG merge 字段流 | 实际 Unit 验证四组 tensor 合并与分支额外字段清空；通过 |
| Unit 依赖声明 | 实际类验证 Face 输出为字符串、图按模型名称精确匹配；通过 |
| Animate 条件处理 | 替身预处理验证负向面部全 -1 和条件从 81 截到 77 帧；通过 |
| Animate 输出裁剪 | 提取真实裁剪条件，21 个 dummy latent 时间槽剩 20 个；77 帧仅按普通 VAE 公式计算，通过，不含实际解码 |
| TeaCache 控制流 | 真实类验证中间残差复用、首末强制计算、拒绝 Wan2.2 标签；通过，不验证近似生成质量 |
| 时间滑窗融合与切片 | 真实 tiler 配合 identity 前向验证重叠融合、仅指定条件切片及参数恢复；通过，einops 的单一 repeat pattern 使用等价 reshape 替身 |
| 滑窗调用参数 | AST 检查专项音频/姿态/音乐和 checkpoint 字段未转发；通过 |

轻量执行环境中 PyTorch 为 `2.8.0`，CUDA 不可用，未安装 PEFT 和 einops。检查按需提取源码类/方法并提供最小依赖，不代表包整体能够在此环境完成导入。未执行完整模型下载、端到端训练、视频生成、LoRA 恢复、多 GPU 训练或加速/量化组合测试。

因此，本文的网络链路、配置和权重约定主要来自源码；数值 scheduler 结论有直接执行检查；显存需求、吞吐、视频质量和组合兼容性不作实测结论。

## 20. 关键概念速查

| 容易混淆的概念 | 当前实现中的区别 |
|---|---|
| `input_video` | SFT 中为监督目标；推理中可作为 Video-to-Video 起点 |
| `input_latents` / `latents` | 干净目标 latent / 当前带噪或采样状态 |
| `noise_pred` / 纯噪声预测 | 标准 SFT 监督的是流速度，不是纯噪声 |
| boundary 索引比例 / switch 阈值 | 调度数组位置比例 / 归一化 timestep |
| 训练 `dit` / 推理 `dit2` | low-noise 单独训练时也叫 `dit`；双模型推理才放到 `dit2` |
| `data_file_keys` / `extra_inputs` | 加载哪些文件 / 传入哪些条件 |
| 普通 I2V / TI2V 首帧融合 | 条件 `y` 拼接 / 直接固定 latent 首帧 |
| `WanModel.forward` / `pipe.model_fn` | 模型类方法 / 当前训练推理选择的实际执行入口 |
| 参数进入字典 / 参数参与计算 | 接口接收成功 / 实际分支消费并影响输出 |
| cache / 原始数据目录 | 三字典 `.pth` / metadata 指向的媒体文件 |
| checkpoint / 完整训练状态 | 可训练参数产物 / 优化器等状态；当前默认只保存前者 |
| LoRA fuse / hotload | 修改基础权重 / 包装层动态应用 |
| `clear_lora` / 恢复原模型 | 清动态列表 / 不撤销融合 |
| Logger step / optimizer update | 微批次计数 / 梯度累积后的实际更新 |
| VAE tiling / DiT 滑窗 | 空间编解码分块 / 模型前向时间分块 |
| Pipeline offload / training offload / ZeRO | 三套不同的参数管理机制 |
| DDP / USP | 数据并行训练 / token 序列并行推理 |
| 输出 FPS / 输入采样 FPS | 视频编码参数 / 训练读取时间间隔 |
| 原始 VAE mean / Pipeline latent | encoder posterior mean / 按通道固定常量归一化后的表示 |
| framewise / 视频 VAE | 独立单帧编码解码 / 带时间缓存的视频编码解码 |
| VACE / TI2V 参考槽监督 | 参考槽进入普通损失 / 首帧专门排除 |
| S2V 音频全部分段 / 一次调用 | 编码器可产生多段 / 普通 Unit 选择第一段 |
| 持续量化 / dequant_once | 量化 Linear / 带量化误差的普通浮点 Linear |
| 模型 train/eval / scheduler.training | 网络模块模式 / 预处理和调度模式 |
| 整模型编译 / 区域编译 | 顶层 forward / 实际重复 block |
| 抽样索引 / 加噪查询索引 | 前者分区采样 / 后者受 timestep dtype 转换及最近网格查找影响 |
| Animate / Animate2 | 姿态与面部适配器 / 专项参考分支和 KV 管理 |
| Fun reference token / sampler 参考 latent 槽 | 网络前向内部加减 / 采样状态时间维增减 |
| TeaCache / 完整预测缓存 | block 残差近似复用 / 不是直接复用 noise_pred |
| cfg_merge / 全条件自动合并 | 四个指定 tensor 名字 / 不通用保留分支条件 |
| WanToDance 音乐 / S2V 音频 | 35 维手工音乐特征 / 学习式语音 encoder 输出 |

这组区别决定了如何理解当前 commit 中的训练产物、条件表示和执行结果。配置名称相似或接口能够接收某个参数，都不能替代实际数据路径和模型分支的语义。
