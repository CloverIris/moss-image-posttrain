# VAE 逐层与层内耗时分析

> 更新：本文测量属于此前设置全局 autotune=0 的历史运行。项目现已移除全局调优和精度覆盖；原生 XLA 默认调优本身也能达到亚秒解码。历史约 65× 只相对受限基线，不能代表相对正常 XLA 的收益。当前策略与复测见 [恢复 XLA 默认优化](xla_defaults.md)。

> 本文记录优化前基线。后续已完成卷积优化，全 FP32 1024 解码约 31.76s → 0.49s，见 [卷积优化与推荐配置](vae_convolution_optimization.md)。

2026-10-20；5070Ti_REMOTE，RTX5070Ti，JAX/jaxlib 0.10.2。
1024×1024，固定原始sample 3（girl seed1052）latent，batch=1。
比较原FP32 decoder与连续FP16 decoder（归一化统计、softmax与必要累加保持FP32）。
使用全局XLA autotune level0，与此前生产VAE保持一致；不改模型结构或参数。

## 测量口径

本报告分别保留两类证据，不混用分母：

1. **完整JIT图GPU trace**：预热、同步后采集，查看完整解码中实际GPU kernel和cuDNN调用的设备执行时间。
   host enqueue/launch时间不当成GPU算子时间；编译、权重加载、图像保存与reward不计入。
   GPU kernel时长总和与完整解码墙钟时间分别报告，重叠、空隙和无法归属的事件保留。
   若一个kernel融合了GN、SiLU、类型转换等多个算子，按融合组归属，不虚构各项精确比例。
2. **真实输入的独立层/子算子微测量**：逐层推进真实activation和该层weights，显式作为JIT参数。
   每层作为整体测量；再分别测层内GN、激活、卷积、shortcut、残差等。
   预热后重复并同步，统计热中位数，编译单列。
   拆分会改变融合和中间张量物化，子算子耗时之和不必等于完整层，更不能替代整图时间。
   attention内部按真实query chunk测量时，乘chunk数属于估算，完整attention另行实测。

占比必须注明“完整图设备kernel总时长”“独立完整层总时长”或“该层独立子算子总时长”。
所有显存峰值与预留9.6GiB容量缓冲区分开说明；预留不是实际DiT。
不同物理GPU、不同运行时段的小幅差异不宜当成可靠速度提升。

## 总体与逐层结果

FP32在GPU1、连续FP16在GPU2，各层首次调用之后同步计时3次，取中位数。每个 `compiled(...) + block_until_ready()` 的时长**包含主机dispatch、启动与同步开销**，不是纯GPU kernel时间；不包含编译、权重加载、图像传输/保存、统计或reward。尤其几十到几百微秒的小算子会受主机开销影响。

|层/模块|输出C×H×W|FP32（ms）|独立层总和占比|连续FP16（ms）|独立层总和占比|
|---|---|---:|---:|---:|---:|
|`bn_unpack`|32×128×128|0.313|0.001%|0.151|0.001%|
|`post_quant_conv`|32×128×128|0.452|0.001%|0.456|0.002%|
|`conv_in`|384×128×128|10.331|0.033%|5.153|0.018%|
|`mid.res0`|384×128×128|247.525|0.789%|242.250|0.841%|
|`mid.attentions.0`|384×128×128|17.850|0.057%|66.144|0.230%|
|`mid.res1`|384×128×128|247.446|0.788%|242.426|0.842%|
|`up0.res0`|384×128×128|247.489|0.789%|242.239|0.841%|
|`up0.res1`|384×128×128|248.714|0.793%|242.811|0.843%|
|`up0.res2`|384×128×128|248.737|0.793%|243.343|0.845%|
|`up0.upsample`|384×256×256|500.112|1.594%|487.943|1.694%|
|`up1.res0`|384×256×256|1000.688|3.189%|976.466|3.390%|
|`up1.res1`|384×256×256|1001.215|3.190%|976.408|3.390%|
|`up1.res2`|384×256×256|1000.972|3.190%|976.558|3.390%|
|`up1.upsample`|384×512×512|4363.282|13.904%|3293.586|11.434%|
|`up2.res0`|192×512×512|3295.475|10.501%|2595.522|9.011%|
|`up2.res1`|192×512×512|1707.033|5.440%|1500.731|5.210%|
|`up2.res2`|192×512×512|1707.723|5.442%|1501.209|5.212%|
|`up2.upsample`|192×1024×1024|5602.046|17.851%|5439.099|18.882%|
|`up3.res0`|96×1024×1024|4429.168|14.114%|4273.013|14.834%|
|`up3.res1`|96×1024×1024|2730.029|8.699%|2727.792|9.470%|
|`up3.res2`|96×1024×1024|2728.421|8.694%|2725.902|9.463%|
|`output`|3×1024×1024|46.884|0.149%|46.189|0.160%|

22层独立调用中位数之和：FP32 **31.38190s**，FP16 **28.80539s**。此数是微测量相加，不是完整图解码时长。

串联结果对各自未拆分decoder原始浮点输出：

|精度|最大绝对误差|相对L2|逐位一致|
|---|---:|---:|---|
|fp32|2.44379044e-06|3.48165313e-07|否|
|full_fp16|0.00146484375|0.000298826204|否|

全部输出有限；拆分JIT会改变融合、舍入与内核选择。**FP32 attention独立层17.85ms不能替代原整图约75ms的矩阵乘耗时，也不能据此说FP32比FP16快。** 两份FP32 attention JAXPR全文相同，但独立层的QK/PV选择64×64/stages3，原整图及FP16为16×8/stages1；共享persistent autotune cache也处于开启状态，不能只归因于JIT边界。见 [attention语义与HLO核查](../reports/vae_fp32_attention_semantics.json)。

## 每层内部与最重算子

以下百分比的分母仅是**该层拆开的子算子同步调用中位数之和**，包含启动/同步开销。它不是整图GPU归因；卷积项包含bias。细分到单个norm1/norm2/silu1/silu2的时间和形状在 [完整CSV](../reports/vae_layer_profile.csv) 与 [JSON](../reports/vae_layer_profile.json)。

### FP32 ResNet内部占比

|ResNet|conv1|conv2|shortcut卷积|两次GN|两次SiLU|残差相加|
|---|---:|---:|---:|---:|---:|---:|
|`mid.res0`|49.703%|49.630%|0.000%|0.204%|0.284%|0.178%|
|`mid.res1`|49.754%|49.651%|0.000%|0.163%|0.179%|0.253%|
|`up0.res0`|49.741%|49.877%|0.000%|0.126%|0.069%|0.187%|
|`up0.res1`|49.645%|49.638%|0.000%|0.244%|0.263%|0.210%|
|`up0.res2`|49.607%|49.515%|0.000%|0.312%|0.363%|0.204%|
|`up1.res0`|49.775%|49.768%|0.000%|0.207%|0.155%|0.094%|
|`up1.res1`|49.758%|49.762%|0.000%|0.229%|0.159%|0.092%|
|`up1.res2`|49.809%|49.755%|0.000%|0.197%|0.144%|0.095%|
|`up2.res0`|66.132%|25.748%|7.886%|0.122%|0.074%|0.038%|
|`up2.res1`|49.795%|49.804%|0.000%|0.192%|0.127%|0.081%|
|`up2.res2`|49.788%|49.809%|0.000%|0.186%|0.131%|0.086%|
|`up3.res0`|63.336%|30.740%|5.605%|0.165%|0.106%|0.048%|
|`up3.res1`|49.788%|49.819%|0.000%|0.191%|0.118%|0.084%|
|`up3.res2`|49.826%|49.769%|0.000%|0.197%|0.128%|0.080%|

|其他模块|主要工作|该层拆分时间占比|其余工作|
|---|---|---:|---|
|`bn_unpack`|`bn_denormalization_fp32`|88.896%|unpack_and_decoder_cast: 11.104%|
|`post_quant_conv`|`conv_including_bias`|100.000%|无|
|`conv_in`|`conv_including_bias`|100.000%|无|
|`up0.upsample`|`upsample_conv_including_bias`|99.951%|nearest_repeat: 0.049%|
|`up1.upsample`|`upsample_conv_including_bias`|99.976%|nearest_repeat: 0.024%|
|`up2.upsample`|`upsample_conv_including_bias`|99.963%|nearest_repeat: 0.037%|
|`output`|`conv_out_including_bias`|91.044%|norm_out: 5.392%；silu_out: 3.564%|

### 连续FP16 ResNet内部占比

|ResNet|conv1|conv2|shortcut卷积|两次GN|两次SiLU|残差相加|
|---|---:|---:|---:|---:|---:|---:|
|`mid.res0`|49.790%|49.742%|0.000%|0.211%|0.197%|0.060%|
|`mid.res1`|49.760%|49.790%|0.000%|0.224%|0.133%|0.093%|
|`up0.res0`|49.768%|49.772%|0.000%|0.243%|0.108%|0.110%|
|`up0.res1`|49.823%|49.759%|0.000%|0.199%|0.147%|0.072%|
|`up0.res2`|49.805%|49.718%|0.000%|0.234%|0.158%|0.085%|
|`up1.res0`|49.890%|49.892%|0.000%|0.103%|0.068%|0.048%|
|`up1.res1`|49.905%|49.911%|0.000%|0.078%|0.052%|0.054%|
|`up1.res2`|49.885%|49.889%|0.000%|0.103%|0.066%|0.057%|
|`up2.res0`|63.072%|28.894%|7.888%|0.079%|0.046%|0.021%|
|`up2.res1`|49.892%|49.889%|0.000%|0.105%|0.071%|0.043%|
|`up2.res2`|49.907%|49.849%|0.000%|0.133%|0.064%|0.046%|
|`up3.res0`|63.591%|31.976%|4.274%|0.086%|0.046%|0.026%|
|`up3.res1`|49.871%|49.929%|0.000%|0.098%|0.062%|0.039%|
|`up3.res2`|49.791%|50.024%|0.000%|0.093%|0.053%|0.039%|

|其他模块|主要工作|该层拆分时间占比|其余工作|
|---|---|---:|---|
|`bn_unpack`|`unpack_and_decoder_cast`|50.583%|bn_denormalization_fp32: 49.417%|
|`post_quant_conv`|`conv_including_bias`|100.000%|无|
|`conv_in`|`conv_including_bias`|100.000%|无|
|`up0.upsample`|`upsample_conv_including_bias`|99.950%|nearest_repeat: 0.050%|
|`up1.upsample`|`upsample_conv_including_bias`|99.982%|nearest_repeat: 0.018%|
|`up2.upsample`|`upsample_conv_including_bias`|99.982%|nearest_repeat: 0.018%|
|`output`|`conv_out_including_bias`|95.670%|norm_out: 2.619%；silu_out: 1.711%|

### attention内部：chunk微测量估算

完整attention包含16,384个token。这里只对真实首个256-query chunk与全部key执行QK、softmax、PV，然后乘64；其余投影测完整张量。**这是拆分执行的估计和，不能相加后替代完整attention实测，更不能作为整图attention内部精确占比。** 该表显示矩阵乘占主导，FP32估算同时明显受kernel选择和多次同步开销影响。

|子算子|FP32估算/实测（ms）|估计和占比|FP16估算/实测（ms）|估计和占比|
|---|---:|---:|---:|---:|
|`group_norm`|0.396|0.257%|0.115|0.135%|
|`qkv_projections`|4.566|2.969%|1.806|2.113%|
|`QK_representative_chunk`（chunk×64估算）|86.683|56.371%|36.812|43.076%|
|`softmax_representative_chunk`（chunk×64估算）|30.855|20.065%|7.132|8.345%|
|`PV_representative_chunk`（chunk×64估算）|29.020|18.872%|38.575|45.139%|
|`output_projection`|1.774|1.154%|0.780|0.913%|
|`residual_add`|0.479|0.312%|0.238|0.279%|

估计和为153.774ms / 85.459ms；独立完整attention实测为17.850ms / 66.144ms，明显不能混用分母。

## 完整图GPU trace核对

实测完整图：GPU3，预热1次后采集1次。下表为 **GPU kernel 时长总和**；FP16 采集开启了 cuDNN debug logging，定位瓶颈可信，小幅性能差异不能当成受控加速结论。

|指标|FP32|连续FP16|
|---|---:|---:|
|kernel 总时长（s）|31.728217|28.916352|
|区间并集（s）|31.728121|28.916304|
|首尾跨度（s）|31.728542|28.916932|
|跨度内空隙（ms）|0.420422|0.627906|
|同步墙钟（s）|31.729804|28.918106|

每种精度均有675个真实GPU kernel，36个卷积全部一一映射到权重名。65个 `memcpy32_post` 是设备复制辅助 kernel，不是主机传输；采样区间内没有 memcpy/memset 引擎事件。kernel 时间重叠仅约0.096ms / 0.047ms。

|实际kernel类别|FP32耗时（ms）|占比|FP16耗时（ms）|占比|
|---|---:|---:|---:|---:|
|卷积，含融合bias及部分residual|31622.558|99.66699%|28835.664|99.72096%|
|attention及投影矩阵乘|74.937|0.23619%|65.313|0.22587%|
|归一化/softmax规约|13.184|0.04155%|6.605|0.02284%|
|其他融合操作：norm/SiLU/残差等|15.648|0.04932%|8.375|0.02896%|
|独立softmax kernel|1.155|0.00364%|0.000|0.00000%|
|布局变换|0.614|0.00193%|0.291|0.00101%|
|设备复制辅助kernel|0.120|0.00038%|0.103|0.00036%|
|未分类kernel|0.000|0.00000%|0.000|0.00000%|

非卷积时间分别为105.659ms与80.688ms，全部分类但未强行分配到具体层。**不能把融合kernel中的GN、SiLU、cast拆成虚假的整图精确耗时。** 下节中的独立子算子计时另有自己的分母。

### 全部36个卷积（实际整图内执行）

以下省略统一前缀 `decoder.`；`upN.resM` 对应 `up_blocks.N.resnets.M`。占比的分母始终是该精度整张图的全部GPU kernel时长。`+R` 表示cuDNN调用还融合了残差side input，时间包含这部分。

|卷积层|输出空间/通道|FP32（ms）|整图占比|FP16（ms）|整图占比|
|---|---|---:|---:|---:|---:|
|`post_quant_conv`|128² / 32→32|0.195|0.001%|0.070|0.000%|
|`conv_in`|128² / 32→384|9.841|0.031%|5.037|0.017%|
|`mid.res0.conv1`|128² / 384→384|124.451|0.392%|121.021|0.419%|
|`mid.res0.conv2` +R|128² / 384→384|124.544|0.393%|121.111|0.419%|
|`mid.res1.conv1`|128² / 384→384|124.708|0.393%|121.041|0.419%|
|`mid.res1.conv2` +R|128² / 384→384|124.283|0.392%|121.095|0.419%|
|`up0.res0.conv1`|128² / 384→384|124.508|0.392%|121.059|0.419%|
|`up0.res0.conv2` +R|128² / 384→384|124.510|0.392%|121.104|0.419%|
|`up0.res1.conv1`|128² / 384→384|124.673|0.393%|121.041|0.419%|
|`up0.res1.conv2` +R|128² / 384→384|124.557|0.393%|121.103|0.419%|
|`up0.res2.conv1`|128² / 384→384|124.505|0.392%|121.037|0.419%|
|`up0.res2.conv2` +R|128² / 384→384|124.704|0.393%|121.112|0.419%|
|`up0.upsample.conv`|256² / 384→384|506.544|1.597%|489.860|1.694%|
|`up1.res0.conv1`|256² / 384→384|504.919|1.591%|489.887|1.694%|
|`up1.res0.conv2` +R|256² / 384→384|504.489|1.590%|490.215|1.695%|
|`up1.res1.conv1`|256² / 384→384|504.466|1.590%|489.956|1.694%|
|`up1.res1.conv2` +R|256² / 384→384|504.869|1.591%|490.291|1.696%|
|`up1.res2.conv1`|256² / 384→384|504.589|1.590%|489.910|1.694%|
|`up1.res2.conv2` +R|256² / 384→384|504.356|1.590%|490.417|1.696%|
|`up1.upsample.conv`|512² / 384→384|4391.264|13.840%|3390.935|11.727%|
|`up2.res0.conv1`|512² / 384→192|2189.750|6.902%|1696.390|5.867%|
|`up2.res0.conv2`|512² / 192→192|871.567|2.747%|770.565|2.665%|
|`up2.res0.conv_shortcut` +R|512² / 384→192|291.708|0.919%|200.889|0.695%|
|`up2.res1.conv1`|512² / 192→192|869.954|2.742%|771.869|2.669%|
|`up2.res1.conv2` +R|512² / 192→192|872.080|2.749%|772.671|2.672%|
|`up2.res2.conv1`|512² / 192→192|872.568|2.750%|770.633|2.665%|
|`up2.res2.conv2` +R|512² / 192→192|873.200|2.752%|771.803|2.669%|
|`up2.upsample.conv`|1024² / 192→192|5630.379|17.746%|5441.075|18.817%|
|`up3.res0.conv1`|1024² / 192→96|2811.786|8.862%|2724.461|9.422%|
|`up3.res0.conv2`|1024² / 96→96|1364.287|4.300%|1332.840|4.609%|
|`up3.res0.conv_shortcut` +R|1024² / 192→96|280.460|0.884%|176.380|0.610%|
|`up3.res1.conv1`|1024² / 96→96|1367.410|4.310%|1332.884|4.609%|
|`up3.res1.conv2` +R|1024² / 96→96|1371.074|4.321%|1324.835|4.582%|
|`up3.res2.conv1`|1024² / 96→96|1364.643|4.301%|1335.472|4.618%|
|`up3.res2.conv2` +R|1024² / 96→96|1367.805|4.311%|1331.845|4.606%|
|`conv_out`|1024² / 96→3|42.912|0.135%|43.751|0.151%|

![36个卷积与算子分类](../reports/vae_layer_profile.png)

## 原因与优化方向

1. **当前瓶颈几乎完全在卷积实现。** 原FP32完整图99.667%的GPU时间在36个卷积；连续FP16仍为99.721%。VAE attention全部矩阵乘仅约75ms/65ms，即使彻底消除也无法解决30秒级解码。
2. **优先看高分辨率上采样后的卷积。** `up2.upsample.conv`（1024²、192→192、3×3）为5.630s；`up1.upsample.conv`（512²、384→384、3×3）为4.391s，合计31.586%整图时间。再看 `up3.res0.conv1` 与 `up2.res0.conv1`；前四个卷积合计47.350%。
3. **精度改变没有解决算法选择。** 实际内核均为 `cudnn::cnn::conv2d_grouped_direct_kernel`；FP16日志36次engine设置及两轮72次执行均为frontend engine 0、FP32 compute，真实输入/权重为FP16。没有记录到 `TENSOR_CORE` numerical note，但缺少note或低吞吐本身不能证明硬件指令类型，也不能把frontend engine 0错称为旧版enum 0的IMPLICIT_GEMM。
4. **下一步应先为重卷积选择合适的cuDNN plan，并约束workspace。** 当前全局autotune=0，原因为此前VAE自动调优会尝试超大workspace。此次仅定位，没有修改生产精度或调优策略；不能直接打开全局level4后假定显存安全。当前安装版本的旧workspace cap flag已删除，没有找到可直接使用的卷积workspace字节上限flag。
5. **计算量并非全貌。** 总计约2.999TMAC（按每MAC两次浮点运算为5.997TFLOP）；最重1024²上采样卷积347.892GMAC，当前仅约0.124TFLOP/s的名义有效吞吐。它是MAC/实测时间推导值，不是硬件计数器，也不能当作Tensor Core利用率。

源码核对绑定安装版本的XLA commit `5a9e73cbd92530cac2ac36f4736a774b2412afe2`：
[autotune=0时禁用cuDNN搜索](https://github.com/openxla/xla/blob/5a9e73cbd92530cac2ac36f4736a774b2412afe2/xla/service/gpu/autotuning/autotuner_pass.cc#L346-L352)；
[由descriptor重建frontend engine](https://github.com/openxla/xla/blob/5a9e73cbd92530cac2ac36f4736a774b2412afe2/xla/stream_executor/cuda/cuda_dnn.cc#L3906-L3947)。
实际日志字段、行号、sha与72次执行详见 [engine审计](../reports/vae_cudnn_engine_audit.json)。


## 复现与原始证据

工作目录全部为服务器 `$CODE_ROOT/moss-art-jax`。主要文件：

- [GPU trace汇总](../reports/vae_gpu_trace.json)、[全部卷积CSV](../reports/vae_gpu_trace_convolutions.csv)。
- [22层及层内微测量](../reports/vae_layer_profile.json)、[MAC与shape映射](../reports/vae_layer_macs.json)。
- [cuDNN engine日志审计](../reports/vae_cudnn_engine_audit.json)、[独立复核](../reports/vae_profile_independent_review.json)。
- 原始trace/HLO/日志在 `outputs/vae_gpu_trace/{fp32,full_fp16}/`；原始微测量在 `outputs/vae_layer_profile/{fp32,full_fp16}/`。大文件保留服务器，未进入Git。
- [逐层测量脚本](../scripts/diagnostics/profile_vae_layers.py)、[整图采集](../scripts/diagnostics/capture_vae_gpu_trace.py)、[整图解析](../scripts/diagnostics/analyze_vae_gpu_trace.py)、[绘图脚本](../scripts/diagnostics/plot_vae_layer_profile.py)。

GPU采集示例（输出需新目录，先确认目标GPU空闲）：

```bash
cd $CODE_ROOT/moss-art-jax
$CONDA_PREFIX/envs/jax_env/bin/python scripts/diagnostics/capture_vae_gpu_trace.py \
  --device 3 --variant fp32 --output outputs/vae_gpu_trace/new_fp32
$CONDA_PREFIX/envs/jax_env/bin/python scripts/diagnostics/profile_vae_layers.py \
  --device 1 --precision fp32 --sample 3 --warm-repeats 3 \
  --latents outputs/rollout_1024_cudnn_tuned/repeat_00/endpoints.npy \
  --reference-raw outputs/vae_precision/fp32_reference/repeat_00/sample-000003.raw.npy \
  --output outputs/vae_layers/new_fp32
```

连续FP16逐层测量将 `--precision` 改为 `float16`，参考图为
`outputs/vae_precision/full_fp16_all/repeat_00/sample-000003.raw.npy`；独立进程运行。
FP16整图采集用 `--variant full_fp16`。需要算法日志时，在启动进程前设
`CUDNN_LOGLEVEL_DBG=3` 与 `CUDNN_LOGDEST_DBG=<existing-output-parent>/cudnn.log`。
只重画已保存报告不占GPU：`$CONDA_PREFIX/envs/jax_env/bin/python scripts/diagnostics/plot_vae_layer_profile.py`。



## 验证记录

- 每种精度22个完整层与120个子算子，MAC精确等于独立权重shape推导；完整CSV共284行。
- tiny CPU两精度流程验证、真实GPU输出有限性、逐层输入/输出形状链、计时中位数和分母复核均通过。
- CUPTI每种675个kernel，36个卷积全部映射；类别总和、原始trace时长和CSV独立重新计算一致。
- 微测量与未拆分模型并非逐位一致。FP16层内独立算子组合相对完整层的最大中间绝对差为0.25（对应大幅度activation，最大相对L2约0.000337）；最终raw误差见上表，不应把它误报为像素差或原模型非有限。
- [测量源码快照](../reports/vae_layer_profile_measured_sources.json)记录了计时口径文字更正：原summary里的“device-only”措辞不准确，当前合并报告明确包含主机派发/同步；没有改变原始时间或重测伪装一致。
- 只新增诊断、证据与文档，没有修改生产模型、vendor、精度默认值或环境依赖。GPU1–3已释放。
