# EMA 参考模式

该模式把输出空间 L2 的参考，从固定 Base 改为缓慢移动的 EMA LoRA teacher。它不会改变 ART 的轨迹缓存、A/B 两个训练窗口、奖励归一化或采样策略。

~~~json
{
  "art": {
    "reference_mode": "ema",
    "reference_ema_decay": 0.99,
    "reference_mse_beta": 1.0
  }
}
~~~

reference_mode 默认为 "base"，保持原有行为。reference_mse_beta 继续控制参考输出 MSE 的系数，EMA 模式不会自动修改它。这里的 1.0 只是延续当前试验设置，不是通用推荐值。

## 更新顺序

设 learner 的 LoRA 参数为 θ，参考参数为 θ̄。一次真正应用的 optimizer 更新如下：

1. 固定当前 teacher θ̄，计算本窗口全部微批次的参考输出。参考输出和 teacher 参数都停止梯度。
2. 使用 ART loss 加 β × MSE(learner 输出, teacher 输出) 更新 learner，得到 θ_new。
3. 执行 θ̄_new = decay × θ̄ + (1 − decay) × θ_new。
4. 下一窗口使用新的 teacher。ART 的 A、B 两次 optimizer 更新分别推进一次 EMA。

微批次与梯度累积不会单独推进 EMA。被跳过的 optimizer 更新不推进 EMA，也不会更新它的计数。首次训练从当前 learner 初始化 teacher，因此第一窗口开始时两者相同，参考 MSE 接近零是预期行为。

decay=0.99 的权重半衰期约为 69 次实际 optimizer 更新；当前每次收集有两次更新，相当于约 34.5 次完整收集。衰减越接近 1，参考移动越慢。允许的范围为 [0, 1)；decay=0 会让 teacher 每次完全追上 learner，使下一窗口开始时的参考项为零。

## 权重、内存和精度

EMA 只维护可训练的 FP32 LoRA 因子，冻结底模由 learner 与 teacher 共享，不复制第二份完整 DiT。对当前 rank48/alpha48 attention LoRA，一份参考参数约增加 198 MB（约 189 MiB）存储；实际值由参数数量决定。

这是优化参数树中 LoRA 因子的 EMA。由于合并增量包含矩阵乘积 A @ B，它不等价于对合并后的全量权重增量逐元素做 EMA。参考前向仍使用与 learner 相同的 BF16 底模、CFG、mask、状态和时间步，并在 learner tape 之外先计算。

阶段切换时，EMA 随 learner checkpoint/handoff 保留；推理和评估仍使用 learner 权重，不会自动改为 EMA 生图。

## 恢复和旧 checkpoint

新 EMA checkpoint 将 teacher 参数、decay 和 EMA 更新次数，与 learner/optimizer 一起原子写入并校验。正常恢复必须带完整 EMA 状态；缺少 teacher、参数 shape/dtype 不匹配或 decay 不匹配都会失败，不会静默重置到当前 learner 或 Base。

从旧 Base-reference checkpoint 切换到 EMA，需要显式迁移：

~~~bash
JAX_PLATFORMS=cpu $CONDA_PREFIX/envs/jax_env/bin/python   scripts/initialize_ema_reference.py --config /absolute/path/migrate_ema.json
~~~

迁移配置包含 source_repo、source_config、target_config、source_checkpoint、target_checkpoint 和 report。迁移从旧 checkpoint 的当前 learner 初始化 teacher，保留 learner、optimizer、update、rollout 和 prompt cursor；EMA 局部计数从 0 开始，来源进度写入迁移记录。源 checkpoint 不修改。目标必须是新目录，算法差异只允许新增 EMA 参考模式及其衰减，不能顺便改变 beta 或学习率。详见工具的配置校验。

当前运行的 Base-reference 训练不会因为代码中新增模式而改变。

## CPU 示例和验证

从本代码目录运行：

~~~bash
JAX_PLATFORMS=cpu JAX_NUM_CPU_DEVICES=2 $CONDA_PREFIX/envs/jax_env/bin/python run.py train --config configs/ema_reference.example.json
~~~

这是 tiny 模型的双逻辑 CPU smoke 配置，用于检查流程，不是实际画质训练。activation_storage="gpu" 在现有配置中表示保留在计算设备；该示例 platform="cpu"，不会使用显卡。

四逻辑 CPU 设备上的集成测试：

~~~bash
JAX_PLATFORMS=cpu JAX_NUM_CPU_DEVICES=4   $CONDA_PREFIX/envs/jax_env/bin/python -m pytest -q   tests/test_reference_state.py tests/test_reference_checkpoint.py   tests/test_ema_reference_runtime.py tests/test_ema_training.py
~~~

测试覆盖 EMA 的 A/B 更新顺序、每窗口冻结、跨阶段保存、完整恢复、跳过更新及重试发布不重复应用更新。CPU tiny 模型验证的是数值和状态流程，不能证明完整模型吞吐或画质收益。

## 指标与判断

EMA 使用 reference_* 指标记录 MSE、加权 loss、loss 占比和输出空间 cotangent 范数比/cosine；reference_mode_ema、reference_ema_decay、reference_ema_updates_used 和 reference_ema_updates 标明本窗口使用的 teacher 版本及更新后的计数。EMA 模式不再用 base_reference_* 名称描述参考项。

输出空间梯度范数比不等于 LoRA 参数梯度比。EMA 允许长期基线漂移，也可能随着错误奖励一起漂移；它不保证解决奖励投机或图片破碎。继续观察固定 heldout prompt、固定 seed 的原始评分和图片，不能用组均值恒定的 rank reward 判断质量提升。

## 精简 TensorBoard

configs/ema_reference_tensorboard.example.json 已将EMA通用train/reference_*映射到03_l2，并显示decay、本窗口teacher版本和更新后计数。该示例对应tiny CPU输出；部署新训练时替换source_log_dir/log_dir。现有Base训练的TensorBoard服务和配置没有修改。

旧检查点迁移配置示例：configs/initialize_ema_reference.example.json。替换其中目标配置及新旧检查点路径；该命令只迁移，不启动训练。服务器副本已复制并校验原有cuDNN bounded卷积二进制到outputs/native，未修改驱动或CUDA。
