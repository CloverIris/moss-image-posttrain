# short/full 条件长度与独立 Qwen prefill

## 先扫描数据集：纯 CPU，无模型推理

```bash
cd $CODE_ROOT/moss-art-jax
$CONDA_PREFIX/envs/jax_env/bin/python -m moss_art.prompt_lengths --config configs/analyze_prompts.json
```

配置示例：

```json
{
  "kind": "sft",
  "manifest": "data/train.jsonl",
  "model": "$MODEL_ROOT/MOSS-Image-SFT",
  "max_reference_pixels": 262144,
  "negative_prompt": "",
  "short": 0,
  "full": 4096,
  "output": "reports/prompt_lengths.json"
}
```

`kind` 取 `sft` 或 `art`，分别读取对应 JSONL。SFT 扫描不要求 target_image，
但填写的图片路径必须存在。只加载 tokenizer、读取图片头中的宽高和 EXIF 方向；
不解码整张图片、不加载 Qwen/VAE 权重、不导入 JAX/Torch/Transformers、不访问 GPU。
图片数不限制；按真实顺序累加每张参考图的视觉 token，并计入文本和模板特殊 token。

缩放规则与当前 Qwen processor 一致：patch=16、merge=2、尺寸对齐32、每图像素预算。
视觉 token 数为 `grid_t * grid_h * grid_w / merge²`。参考图的原始尺寸不能直接用于
这个公式，必须先计算缩放后的 grid。当前脚本仅支持静态图片，未接视频。

输出报告包含每行的 token 数、各参考图 token 数/grid、负条件长度、P50/P70/P90/P95/P99、
建议 short、总样本数、保留数、丢弃数、short 的实际覆盖率。
`short=0` 对 full 内的有效请求取最近秩第70百分位，再向上对齐64；确保至少覆盖
70%的**保留请求**。同时报告所有输入请求的覆盖率；超长丢弃较多时，这个比例可能不足70%。
显式填写 short 时只报告实际比例，不虚报覆盖率。若分布集中，short 与 full 可以相同。

扫描不修改原 JSONL。超过 full 的条目会打印 `prompt_dropped`，并写入报告及旁边的
`*.dropped.jsonl`。即使全部超长，扫描仍输出完整报告；正式预处理/训练没有可用样本则报错。

## 使用扫描结果

ART 与 SFT 训练/采样均采用：

```json
{"data":{"short_context_length":512,"context_length":4096}}
```

`context_length` 就是 full/max length。`short_context_length=0` 也可在启动时按实际
缓存数据自动选择，结果会记录日志。SFT 离线预处理对应：

```json
{"short_tokens":512,"max_tokens":4096,"device":4}
```

先计数，再选择 short/full 容量。`length == full` 保留，`length > full` 丢弃并记录ID、
真实长度、full及阶段，不截断、不会在每个 epoch 重复消耗无效样本。
ART 正负条件任一超长就丢弃整个请求。旧缓存中超长的 SFT 样本在 coordinator/worker
以相同顺序过滤，之后才产生 sample index 和随机种子。

DiT 排列仍为 `[生成图片 token][有效条件 token][padding]`。cuDNN attention 使用真实
sequence length，不建立稠密 GPU mask。短长两个桶分别编译；同桶内实际 prompt 长度
变化不重新编译。分辨率、microbatch、dtype、LoRA rank 等其他 shape/config 变化仍可能
产生新编译。SFT microbatch 遇到不同分辨率或不同桶会提前结束当前 batch，保持样本/seed
顺序；梯度仍按全局样本数归一化。

## GPU 4 上只做一次 Qwen prefill

物理 GPU 4 是第五张卡，GB206(SM120) 16GB；运行时按 UUID 隔离，子进程内显示 CUDA 0。
GPU 0~3 留给 DiT。ART 的 `preparation.device` 和 SFT 准备配置的 `device` 默认均为4。

ART：维护中的配置已开启 `preparation.async_prefetch=true`，使用两个共享内存缓冲，
GPU4 的下一批 prefill 与当前 rollout/update 重叠，见 ASYNC_PREFETCH.md。
关闭异步时，`preparation.auto_prepare=true` 使 train 启动 DiT workers 前调用独立 JAX 预处理
子进程，把缺少的条件编码到磁盘。也可手动运行 `prepare-text --config ...`。
完整缓存再次准备时 `encoded_now=0`，不加载 Qwen 权重；所有 GPU、每个 seed、每轮 rollout
和两次 ART 回放更新读取同一份缓存特征。没有随 LoRA 更新重新编码冻结 Qwen。
离线整库预处理完成后进程退出、释放 GPU4；异步模式则在本次训练期间保持生产者存活。
两种模式都不是对外 HTTP 服务。

SFT：CPU 沿用官方 processor，Qwen 模型推理由独立 JAX 子进程完成，包括视觉塔与
多模态 MRoPE。相同 prompt + 有序参考图内容共享一个条件文件，目标图不同也能复用。
预处理 cache identity 绑定模型、tokenizer、processor、图片内容、设置与编码实现；
不同身份不能混写。修改这些内容后使用新 cache 目录。
Qwen 子进程退出后，父进程才在该卡加载 Torch FP32 VAE 编码训练目标。

Qwen 只计算 prefill 的最后隐藏状态，不运行 LM head、逐 token 生成或维护 KV cache。
语言层 BF16；full-attention 层使用 cuDNN attention，gated-delta 层用原生 JAX。
图像条件的视觉塔采用 FP32 JAX 计算（FP32 attention 使用 XLA），输出在语言模型入口转
BF16。这是对照测试发现 BF16 视觉误差放大后的保守选择，并非宣称与原 BF16 Torch
输出逐位一致。DiT 的 BF16 cuDNN attention 不受影响。
权重保持 host mmap、逐层传入 GPU，避免8B编码器完整驻留16GB显存；固定 short/full
长度复用已编译层。没有任何全局 XLA autotune/优化限制。

ART 仍是文本条件入口；多参考图目前接在 SFT/对应采样，未扩展为多图 ART 奖励数据入口。
