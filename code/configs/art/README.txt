DiffusionART 官方对齐基线（MOSS / JAX / K=4）
官方基准：85ac6f190c9f1ce22f96350ba2655ae310e89afe

核心目标与两次更新语义已经对齐。本次保留MOSS原生调度网格、CFG3、shift5、1024分辨率；
不宣称SD3或论文历史实验逐位复现。官方SD3 paper candidate为10 Euler steps、
K18、64 prompts、AdamW lr3e-4、rank32/alpha64、1000 updates。
当前基线用K4、4 prompts、AdamW lr1e-4、100 updates作为保守起点。
全部10个采样时刻参与回放，两个窗口各取每prompt的2条轨迹。

服务器目录：$CODE_ROOT/moss-art-jax
本机可跑的机制验证/参数模板：
  bash scripts/run.sh train --config configs/art/moss_k4_aligned.json
使用自定义HTTP reward：
  bash scripts/run.sh train --config configs/art/moss_k4_aligned_http.json
训练后的标准50步评估：
  bash scripts/run.sh evaluate --config configs/art/moss_k4_eval50.json
小链路测试：
  bash scripts/run.sh train --config configs/art/moss_k4_aligned_1024_smoke.json

正式训练前需要替换data.prompts。当前configs/prompts.jsonl只有4条示例；
SigLIP cosine仅验证流程，不能代表你实际要优化的画质reward。
HTTP配置中的127.0.0.1:8765/reward沿用接口示例，请指向真正的reward服务。
eval50配置通过run.adapter导入导出的LoRA；若训练HTTP版，改其adapter路径与reward。
评估请使用独立验证集；当前同一示例文件上的在线10步指标仅是训练监测。

每轮总16张，每GPU负责1个prompt、串行生成4张。每window全局80个state/time训练项；
训练microbatch仍为1，K4不等于每卡batch4。
100 optimizer updates = 50个有效rollout = 800张训练图；无信号轮会额外生成。
ART rollout和replay都用BF16，无MXFP8变量。LoRA master与优化器为FP32。
本次把ART的GPU激活检查点设置接线补齐，推荐gpu/每6层重计算；
VAE FP32、cuDNN bounded workspace256MiB，未设置全局XLA优化限制。

旧JSON默认行为保留。源码与配置身份指纹变化，历史checkpoint不能保证直接resume；
要恢复旧optimizer状态需使用原源码，或从导出的adapter新开训练。
