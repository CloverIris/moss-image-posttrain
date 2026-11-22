
## 提示词处理约定

用户提供的是完整的 MOSS Image 层提示词，必须透传，不做语义解释或格式整理。
- 不删除、不插入、不重排 Prompt / Negative Prompt 等标签、自然语言、Tag 或 Layout JSON 正文。
- 不根据正文中的标签拆分正负提示词，也不自动补充质量词或负面词。
- 仅执行约定的一次转义解析：反斜杠+n 转换成换行，两个反斜杠转换成一个字面反斜杠；未知转义保留。
- 然后由编码器添加官方 Qwen chat template。不要要求用户手写 Qwen 特殊标记。
- JSONL 的显式 negative_prompt 字段、配置的 data.negative_prompt 仍是独立 CFG 条件；正文内容不改变它们。
- 不修改用户的原始提示词文件，除非用户明确要求编辑该文件。

## MOSS ART 训练精度约定（用户于2026-10-31明确指定）

- 后续MOSS ART训练中，用户说“FP8”或“MXFP8”一律指 `mxfp8_calibrated`。
- 新建、重试、恢复或派生训练配置时，FP8 rollout必须显式使用 `sampling.rollout_mode = "mxfp8_calibrated"`；禁止使用未校准的纯FP8/MXFP8 rollout进行训练。
- 启动前核对最终生效配置和日志，不得因旧模板或默认值退回 `mxfp8`。BF16训练前向与现有校准机制保持一致。
- 这是训练约定；独立纯推理/评估的精度选择仍按相应任务配置。不要批量改写历史冻结run或正在运行服务的配置。
- 启动后续训练前读取 `$MOSS_ROOT/training_precision_policy.json` 和 `datasets/next_training_dataset.json`。

## MOSS ART 当前开训门禁（2026-10-31用户要求停训）

- 用户要求当前训练结束，只有收到用户新的明确开训通知后才可启动训练；不得自动恢复、续跑或让旧巡检触发训练。
- 下次按用户通知从Base、step0、新optimizer、新run重新开始，不resume已停止轮次。
- 停止轮次：`rank_sqrt_kawai_deepghs_w1_1_top1975v2_P24_K8_T4_1024_U200_calibrated_nol2_20261031_213725`；已完成optimizer更新0次，原图/评分/失败日志均保留。
- 下次开训前先修复并回归验证reward wrapper动态加载导致的`sdg_parallel`延迟导入问题；禁止在等待通知期间试启动训练。
- 启动前读取`$MOSS_ROOT/datasets/next_training_dataset.json`的`training_launch_control`；继续遵守FP8/MXFP8训练均为`mxfp8_calibrated`的约定。

