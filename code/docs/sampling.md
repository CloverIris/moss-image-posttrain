# 50 步 res_multistep + beta

当前默认：`steps: 50`、`sampler: res_multistep`、`scheduler: beta`。用户提到的 `resmutistep` 对应规范名称 `res_multistep`。生产采样与 ART 训练回放使用同一套实现。

## 使用

```yaml
sampling:
  steps: 50
  sampler: res_multistep
  scheduler: beta
  beta_alpha: 0.6
  beta_beta: 0.6
  shift: 5.0
  train_timesteps: [0, 10, 20, 30, 40]
```

当前单卡 1024 训练配置为 `configs/moss_train_1024.json`，计时配置为 `configs/moss_rollout_1024_cudnn_tuned.json`。四卡 `moss_lora_4gpu.yaml` 和 `moss_http_4gpu.yaml` 也采用此默认值；四卡模板的默认分辨率仍为 256，1024 配置单独显式设置。低步数 smoke 和旧后端对照配置显式保留 `euler + moss`，用于既有回归测试。

```bash
cd $CODE_ROOT/moss-art-jax
$CONDA_PREFIX/envs/jax_env/bin/python scripts/benchmark_rollout.py \
  --config configs/moss_rollout_1024_cudnn_tuned.json \
  --adapter outputs/moss_siglip_4gpu/adapter.safetensors \
  --prompt-ids cat,girl --repeats 2 --output outputs/my_res_multistep_beta_benchmark

# 参数已写入下方 JSON；运行前检查数据、输出目录和可选 reward URL。
bash scripts/run.sh train --config configs/examples/sampling-1.json
```

以上命令要求对应设备空闲。需要 HTTP 奖励，在 JSON 中设置 `reward.kind="http"`、`reward.decode=true`、`reward.url="http://REWARD_HOST:8765/reward"`。完整检查点绑定采样器、调度器、参数和源码；旧实验的独立 LoRA 可以作为新实验起点，旧优化器检查点不跨此次变更强行恢复。

## 标准行为与来源

核对并固定 [ComfyUI b1a247c9](https://github.com/Comfy-Org/ComfyUI/tree/b1a247c9cf9dbfda4b2c0a0c1d50950667eeaa0c)。MOSS 在该实现中使用 `ModelSamplingDiscreteFlow + CONST`，不是 Flux 的时间表；MOSS 参数为 multiplier=1、shift=5、1000点。生成器输入速度为 `v`，对应 denoised 为 `d=x-sigma*v`，模型实际时间输入仍为 `1000*sigma`。

Beta 调度：构造升序 `u=j/1000, j=1..1000`，仅一次应用 `sigma=shift*u/(1+(shift-1)*u)`。将 `Beta(alpha,beta).ppf(1-k/steps)` 乘999并用 `numpy.rint` 取最近整数（半整数取偶）作为模型网格索引，删除相邻重复索引，最后附加零。默认 alpha=beta=0.6；50个请求步恰好保留50次模型预测。其他极端参数/步数可能因网格去重减少实际 NFE，日志记录实际 sigma 序列和步数，回放索引超出实际网格时明确报错。`training_shift` 只属于旧 `moss` 调度器；在 beta 配置中必须保持默认1，不对 beta 施加第二次吸附或位移。

`res_multistep` 是确定性 RES 多步方法（eta=0，非 CFG++）。正 sigma 内使用 `t=-log(sigma)`；首步没有历史，末步落到零，这两处使用 Euler。中间步：

```text
h = log(sigma_i / sigma_next)
c2 = -log(sigma_prev / sigma_i) / h
phi1 = expm1(-h) / (-h)
phi2 = (phi1 - 1) / (-h)
b2 = phi2 / c2
b1 = phi1 - b2
x_next = exp(-h)*x + h*(b1*d_current + b2*d_previous)
```

实现使用稳定 phi 函数处理小 h；FP32 latent/velocity/history。每张图、每次 rollout 重置 history，仅保留上一份 denoised，不增加网络评估次数。50 steps 指50次 CFG velocity预测；CFG=3时每次包括独立正、负分支。

来源：[RES 实现](https://github.com/Comfy-Org/ComfyUI/blob/b1a247c9cf9dbfda4b2c0a0c1d50950667eeaa0c/comfy/k_diffusion/sampling.py)、[beta scheduler](https://github.com/Comfy-Org/ComfyUI/blob/b1a247c9cf9dbfda4b2c0a0c1d50950667eeaa0c/comfy/samplers.py)、[flow 模型采样](https://github.com/Comfy-Org/ComfyUI/blob/b1a247c9cf9dbfda4b2c0a0c1d50950667eeaa0c/comfy/model_sampling.py)、[MOSS 参数](https://github.com/Comfy-Org/ComfyUI/blob/b1a247c9cf9dbfda4b2c0a0c1d50950667eeaa0c/comfy/supported_models.py)。算法背景：[RES](https://arxiv.org/abs/2308.02157)、[Beta Sampling](https://arxiv.org/abs/2407.12173)。

## ART 多步回放

缓存仍保存选中状态和该处的原始旧 velocity，不能用混合导数代替。中间步对当前 velocity 的局部系数为 `C=-sigma*h*b1`，首末 Euler 为 `C=sigma_next-sigma`。整组系数随原 rollout 冻结，第二个训练窗口不按新权重重建轨迹。

`art_loss` 通过已有 `ode_velocity_coefficient` 入口接收 C；冻结状态和历史时，旧/新局部 ODE 均值差中的历史项相消，因此无需在 replay cache 中额外保存所有历史 denoised。此扩展把原来的 Euler scheduler-reference 漂移约束适配到多步采样。它仍是局部代理度量，不是整个多步轨迹的精确 KL；`art.noise_level` 不代表采样时增加随机噪声。

cuDNN FlashAttention、BF16 DiT、全 FP32 原生 XLA VAE 保持启用。不设置限制 XLA 调优的全局变量。


## 数值验证范围

独立 CPU 对照从固定版本的本机 ComfyUI/Kohaku 源文件提取原函数，在隔离的 Torch 子进程执行；生产采样不导入 Torch。14种时间表（1/2/10/50/200/1000步、不同shift和beta形状、旧MOSS网格）全部逐位一致，9个常规求解器用例通过。默认 beta50 的非线性 CFG 场逐步最大差为 5.3644e-7，JIT终点最大差为4.1723e-7；constant/affine场的50步最大差不超过8.35e-7。当前velocity局部系数的JVP最大误差7.45e-9，冻结history的JVP为零。这是求解器数值兼容验证，不是两个完整模型的图像逐位一致性声明。

额外人工压力网格 `[1,.9999999,.9999997,.999,.5,0]` 的跨框架最大差6.3539e-5，超过预设3e-5阈值，原失败记录保留。相对独立Float64原公式，Torch32/JAX32最大轨迹误差分别3.3461e-5和6.8860e-5；该例中JAX32更差，不能声称局部phi级数改善整个轨迹精度。默认beta50最小内部h=0.00121299，没有进入h<1e-3分支；默认50步的兼容性结论不依赖这一极端网格。

[独立对照报告](../reports/sampling_reference_validation.json) · [初次严格压力比较](../reports/sampling_reference_validation_initial.json)

```bash
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=2 \
  $CONDA_PREFIX/envs/jax_env/bin/python scripts/diagnostics/benchmark_sampling.py
```


## 真实 MOSS 1024 验证

服务器 `5070Ti_REMOTE`、RTX5070Ti 16GB，BF16 DiT + cuDNN FlashAttention、全FP32 XLA VAE、CFG3、rank4。使用此前独立LoRA初始化新实验，文本特征来自既有缓存。CPU完整回归 **303 tests + 74 subtests** 通过（107.58秒）；含独立回放系数、history重置、mask阈值和连续训练/续训端到端测试。

单卡完成50步、girl/K2、两窗口各5个回放时间步、两次优化器更新。梯度范数分别0.02630、0.06709，全部回放项活跃、未被mask，JAX活跃峰值11.659GiB。此短验证证明流程可运行，不用于宣称长期训练收益或画质提升。


训练检查点通过 COMMITTED、manifest、payload 校验；336个LoRA参数张量与导出safetensors逐字节一致，全部较初始化参数发生变化，673个优化器叶均有限，Adam count=2。总运行230.26秒（含加载、生成、解码、奖励、反向和保存），第二更新窗口36.52秒；不能将首次窗口累计时间当作纯反向耗时。见 [真实训练独立审计](../reports/res_multistep_beta_training_review.json)。

训练产物：`outputs/train_1024_res_multistep_beta_50/checkpoints/step_00000002` 与 `outputs/train_1024_res_multistep_beta_50/adapter.safetensors`。这次使用真实SigLIP奖励，HTTP奖励协议与先前一致。

上游来源核验： [固定文件身份与许可](../reports/sampling_upstream_identity.json)。


## 1024 生图速度

GPU1，两个提示词cat/girl、四个固定seed（42/43/1051/1052），逐图batch1。相同样本重复两轮，第二轮为进程内热运行；四张图的latent、PNG、奖励在两轮间逐位一致。

| 阶段 | 首轮每张 | 热运行每张 |
| --- | ---: | ---: |
| DiT 50步 rollout | 66.965s | **66.070s** |
| FP32 VAE 解码 | 0.452s | **0.429s** |
| **生图合计** | 67.416s | **66.499s** |
| 含CPU SigLIP奖励 | 70.802s | 69.563s |

生图约0.90张/分钟；此路径保留ART缓存，计时包含同步、选中latent/velocity回传，使用已缓存文本，不含在线文本编码、模型首次加载、PNG写盘和奖励。两轮只包含四张独立图；性能统计不构成画质评测。首轮与GPU2训练及CPU回归重叠，热轮在它们结束后执行。

JAX进程活跃峰值 **11.659GiB**，nvidia-smi另记录分配器保留显存；单卡16GB完成，无OOM，不需要两卡FSDP。GPU验证结束后进程已退出释放设备。过程环境确认没有XLA_FLAGS、TF_XLA_FLAGS、TF_FP16_CONV_USE_FP32_COMPUTE或JIT禁用覆盖。

[汇总](../reports/res_multistep_beta_validation.json) · [生图独立审计](../reports/res_multistep_beta_generation_review.json) · [原始benchmark](../outputs/generation_res_multistep_beta_50/benchmark.json) · [四图画廊](../outputs/generation_res_multistep_beta_50/repeat_01/index.html)
