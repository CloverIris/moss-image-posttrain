# 奖励器与扩展协议

## HTTP 远程奖励

使用 `reward.kind: http` 后，训练和固定 seed 评估都会把已由 JAX VAE 解码的 PNG 与提示词，通过 HTTP POST 发给指定 URL。服务只需返回分数，越高表示越好。请求不包含模型权重、LoRA、优化器状态或服务器本地文件路径。

已有符合下述协议的服务时，直接传 URL：

```bash
bash scripts/run.sh train --config configs/moss_http_4gpu.json
```

在 JSON 中设置 `reward.kind="http"`、`reward.decode=true`、`reward.url`。
初始适配器写入 `run.init_adapter`，输出目录写入 `runtime.output_dir`，更新数写入
`training.max_updates`。URL 需要指向训练服务器可访问的评分服务。完整配置方式如下：

```yaml
reward:
  kind: http
  url: http://127.0.0.1:8765/reward
  decode: true
  batch_size: 1
  timeout: 120
  http_max_retries: 2
  http_retry_backoff: 0.5
  http_bearer_token_env: ""
  http_response_limit_bytes: 1048576
```

`batch_size` 控制一次 POST 的图片数量；默认 1，兼容只返回单个 reward 的接口。提高后，服务需要返回同等数量的数组。图片评分全部完成后才进入该轮 ART 更新，无法评分时不会替换为零分。

### 请求与返回格式

请求头为 `Content-Type: application/json`、`Accept: application/json`。请求体是 UTF-8 JSON：

```json
{
  "version": 1,
  "items": [
    {
      "index": 0,
      "prompt": "a red bird",
      "image_base64": "<PNG 文件字节的标准 Base64，不带 data: 前缀>",
      "mime_type": "image/png"
    }
  ]
}
```

`index` 对应这一次完整评分调用中的图片位置，分批 POST 时不从零重新计数；它不是整个训练任务的唯一编号。服务须按请求中 `items` 的顺序返回结果，不能按自身排序返回。

推荐响应：

```json
{"rewards": [0.73]}
```

也接受 `{"scores": [0.73]}`。请求只有一张图时，还接受 `{"reward": 0.73}` 或 JSON 数字 `0.73`。一个对象只能包含 `rewards`、`scores`、`reward` 中的一个字段；可以包含其他服务元数据，但当前客户端只使用分数。分数必须是有限数值且可表示为 FP32；字符串、布尔值、NaN/Inf、嵌套数组、数量不符都会报错。

返回 HTTP 2xx 且符合格式才算成功。429、502、503、504 和连接/读写失败可重试；默认最多额外重试 2 次，等待 0.5、1 秒。等待按指数增长，单次上限 30 秒。参数 `timeout` 是连接/读写等待超时（socket I/O timeout），不是整个请求或包含重试的总墙钟时限；持续分段返回数据可能超过该数值，DNS 查询也不受严格的总时限保证。其他 HTTP 错误、证书校验失败或不合规响应直接失败；不跟随重定向。默认最多读取 1MiB 响应，避免错误服务返回无限大内容。

同一批次的重试沿用相同请求体和 `Idempotency-Key`，不同批次生成新 key。网络超时不代表远端没有执行；有计费或写入行为的服务需自行按该 key 去重。示例服务仅做评分，不保存去重状态。

### 可选鉴权与版本固定

如果服务需要 Bearer token，将令牌放在训练进程继承的环境变量中：

```bash
export MOSS_REWARD_TOKEN='<你的令牌>'
```

然后配置：

```yaml
reward:
  kind: http
  url: https://reward.example.com/v1/reward
  http_bearer_token_env: MOSS_REWARD_TOKEN
```

对于上述 Bearer 鉴权，只保存环境变量名，不把该环境变量中的 token 值写入配置或检查点。URL 会原样保存到配置和检查点身份中，不要在 URL 查询参数中嵌入凭据；使用环境变量 Bearer 鉴权。环境变量缺失或含无效字符会在发送请求前失败。URL 不接受用户名密码形式；网络传输中要保护图片、提示词与令牌时使用 HTTPS。

检查点绑定 URL 和奖励配置，不能自动证明远端模型、预处理和代码未变。评分服务应提供稳定的模型版本入口并保持续训期间一致；更换服务版本、URL 或其他奖励设置时，用 JSON 中的 `run.init_adapter` 开始新实验。更新本工程源码也会改变精确续训身份。

### 可运行的服务示例

[examples/http_reward_server.py](../examples/http_reward_server.py) 使用 Python 标准库 HTTP 服务，默认仅监听本机 `127.0.0.1:8765`，不依赖 Flask/FastAPI。先用亮度 toy 奖励检查传输：

```bash
cd $CODE_ROOT/moss-art-jax
CUDA_VISIBLE_DEVICES="" $CONDA_PREFIX/envs/jax_env/bin/python examples/http_reward_server.py --kind toy
```

toy 奖励只是图片平均亮度，不能用于正式画质训练。需要真实图文语义分数时，在评分机器上使用已有本地 SigLIP2 与独立 CPU 环境：

```bash
CUDA_VISIBLE_DEVICES="" $CONDA_PREFIX/envs/jax_env/bin/python examples/http_reward_server.py \
  --kind siglip --host 127.0.0.1 --port 8765
```

跨机器访问时用 `--host` 指定评分机器的可达监听地址（如 `0.0.0.0`），客户端 URL 则填写评分机器实际 IP/域名。可加 `--bearer-token-env MOSS_REWARD_TOKEN`，服务端与客户端设置相同 token。该示例不直接实现 TLS；正式部署可接已有网关或换成自己的评分服务。

示例的 `--model-path`、`--python`、`--threads`、`--batch-size` 可配置 CPU SigLIP。它沿用现有隔离子进程实现，每个请求会重新加载奖励模型，便于验证，吞吐量有限。默认同时处理一个评分请求，其余返回 429；可用 `--max-concurrent` 调整。

服务验证协议版本、图片编码、PNG 文件与批量数量，并限制请求体（默认 16MiB）、每图像素数（默认 16M）与条目数（默认 64）；同批图片需具有相同尺寸。错误返回简短 JSON，不返回评分子进程的原始 stderr。使用 `--help` 查看可调整的限额。此示例适合接入验证；已有评分模型可替换服务端的 `score_images` 调用，保持协议即可。

## 内置 SigLIP2

默认 `reward.kind: siglip` 使用本地 `google/siglip2-so400m-patch14-384` 快照，外部 `torch_env` Python 在 `torch.inference_mode()` 下运行。默认只允许 CPU；不与训练 worker 竞争 GPU。`local_files_only=True`、`trust_remote_code=False`，不会在训练时下载模型或执行仓库远程代码。

```yaml
reward:
  kind: siglip
  decode: true
  cpu_decode: false
  model_path: $HOME/.cache/huggingface/hub/models--google--siglip2-so400m-patch14-384/snapshots/e8e487298228002f3d8a82e0cd5c8ea9c567f57f
  python: $CONDA_PREFIX/envs/torch_env/bin/python
  device: cpu
  batch_size: 1
  metric: cosine
  threads: 4
  timeout: 900
```

`cosine` 为匹配图片和提示词归一化 embedding 的点积；`logit` 返回对应成对 logit。二者都以较高为较好。每次评分通过临时 PNG/JSON 启动独立进程，默认会重新加载模型；代价是额外 CPU 时间，但不保留跨框架训练状态。本机单图奖励 smoke 含模型加载约 10 秒，不能视为任何批量或图片的固定延迟。

SigLIP2 自己的文本长度上限与 MOSS 编码器不同；长描述可能在奖励端被截断。默认奖励衡量文本语义匹配，不专门衡量审美、角色身份、画面文字正确性或人体结构。在线优化可能利用奖励器的偏差，需检查实际图像。

## 外部 command 协议

这是接入本地评分模型或自定义外部程序的入口；HTTP 服务可直接使用上面的内置 HTTP 后端。训练模型、LoRA 与优化器仍是 JAX；外部程序只返回标量。

```yaml
reward:
  kind: command
  decode: true
  command: [/absolute/path/python, /absolute/path/scorer.py]
  timeout: 900
```

默认实际执行：

```text
/absolute/path/python /absolute/path/scorer.py --request /tmp/.../request.json --output /tmp/.../result.json
```

也可显式在 argv 中放 `{request}`、`{output}` 占位符。执行使用 `shell=False`，不会解释 shell 管道或重定向。

请求协议为 UTF-8 JSON：

```json
{
  "version": 1,
  "items": [
    {"index": 0, "prompt": "a red bird", "image": "/tmp/.../image-000000.png"},
    {"index": 1, "prompt": "a blue bird", "image": "/tmp/.../image-000001.png"}
  ],
  "config": {}
}
```

程序必须在 `--output` 指定的文件中写出：

```json
{"scores": [0.7, 0.2]}
```

顺序必须与请求相同，数量必须精确对应，值必须是有限标量。较高表示较好；如果模型输出“错误率”，应在包装器中取负或做适当变换。非零退出、超时、遗漏输出、NaN/Inf 或分数数量错误都终止当前训练，不会降级为假分数。临时文件在调用完成后清理，长期保存需包装器自行复制到自己的结果目录。

模板可以如下实现；此处没有内置虚构评分，`actual_model_score` 必须连接真实评分逻辑：

```python
import argparse, json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--request', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
request = json.loads(Path(a.request).read_text(encoding='utf-8'))
assert request['version'] == 1
scores = [actual_model_score(item['image'], item['prompt']) for item in request['items']]
Path(a.output).write_text(json.dumps({'scores': scores}, allow_nan=False), encoding='utf-8')
```

自定义程序如需 GPU，须自行显式安排设备和生命周期。协调器默认隐藏 GPU；不能假设 `cuda:0` 自动可见。不要抢占其他任务或把图像评分器放到已接近显存上限的 DiT 卡上。

外部命令的 argv 被配置身份绑定，但无法自动校验其远程服务、容器、可执行程序依赖或远端模型的实际状态。调用方应固定模型版本，记录评分服务版本并保证续训时一致。

## 测试奖励

`toy` 返回 RGB 平均亮度；优化它会趋向白图。`toy_latent` 返回 latent 距固定常数的负均方误差。二者可稳定验证“采样→评分→方向→梯度→更新”，但不能验证图像质量。真实 MOSS 的 `toy_latent` 配置最多允许四次更新，以限制把 smoke 误当成长训练的用法。
