# MOSS BA 少步蒸馏

第一轮在自有服务器上训练 JAX/Flax SideNet。教师是冻结的 MOSS-Image-Base。教师参考、在线监督和少步评估均使用 BF16 全量 DiT，1024 方图，MOSS 原始 shift 5 时间表，CFG 3.0，负分支跳过第 22 层。每次主干调用包含正负两个分支。

输入来自新版 2000 条 prompt。原数据有 5 条超长、4 条固定评估、16 条额外留出和 4 条校准验证。独立训练文件包含剩余 1971 条。训练前检查文件、集合关系与缓存 manifest，正负文本复用已有 Qwen token 特征。第一轮每条训练 prompt 生成一个 48 步教师终点，训练缓存按 FP16 存盘，监督、积分和误差统计用 FP32，留出终点按 FP32 存盘。

SideNet 共享线性层将每个 Qwen token 从 4096 维压至 128 维，保留掩码与正负分支身份。空间路径结合深度卷积、条件调制、降采样的自注意力及文本交叉注意力。输出为随带符号步长缩放的速度残差，末层从零初始化。第一版专门拟合固定 CFG 后的速度场。

每条 prompt 做 4 次更新。每次从教师终点和新噪声构造初始状态，保持学生参数固定走 4 个带正负方向的区间。学生三点 Gauss–Legendre 预测下一状态，冻结教师在该状态给出目标速度并复用为下一锚点。4 项残差 MSE 梯度求均值后更新一次。为覆盖 MOSS 的非均匀时间表，训练间隔混合了截断指数采样和 8／12 步部署区间尺度。优化器为 AdamW，学习率预热至 0.0003。

推理沿用官方双锚点的前向预测、后向探测、四点 Gauss–Lobatto 校正与末段直接返回。JAX 核心在 1、5、8、12 NFE 上经过官方 PyTorch 小张量 oracle 对照。原论文与 MIT 参考实现在 [BA-solver](https://github.com/HKUST-LongGroup/BA-solver/tree/c03aa8a92d7b98c419ba648175ed7a9ee6997829) 和 [论文](https://arxiv.org/abs/2601.21542)。

服务器代码目录为 `$CODE_ROOT/moss-ba-jax`，产物在 `$MOSS_ROOT/distillation/ba_cnn_att_v2_1024_cfg3_20261103`。第 112 次更新后，GPU 0–3 各自生成 48 步 BF16 教师终点，GPU 4（GB206(SM120)）单独更新学生。每条 prompt 完成后，latest.msgpack 原子提交学生、优化器及游标。部署记录在 `pool/deployment.json`。每次更新还需对学生当前状态求教师速度。GPU 4 与 GPU 0 对相同输入实测，数值和延迟均通过后才向 GPU 0 请求在线速度，结果见 `pool/target_benchmark.json`。

在服务器代码目录使用 jax_env 运行 `python pool_status.py --record` 可查看并保存五卡分工、参考积压、实际吞吐和误差。固定留出使用 4 个隔离 prompt 各一个固定 seed，在更新 4、32、128、512、2048 和终点分别对照 48 步教师、等 NFE Euler 以及置零 SideNet 的 BA。误差是 latent MSE，图片按同一 VAE 配对解码。

正常使用可在代码目录运行 `python report.py` 读取进度。在空闲 GPU 上从导出的学生权重做独立对照可执行以下命令（请使用 jax_env，输出目录必须新建）。

```sh
python infer.py --checkpoint /absolute/path/side_000032.msgpack --prompt-id 2070310815398495946 --seed 424242 --steps 8 --gpu 3 --compare --output /absolute/path/new_comparison
```

`result.json` 的计时覆盖去噪并同步到主机。模型加载、Qwen 编码与 VAE 解码独立于该计时。正式判断应结合独立留出和多次计时。Kimi Delta Attention 留作后续独立消融。

