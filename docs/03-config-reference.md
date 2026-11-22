# 配置参考

所有训练、采样、评估、预处理参数都从 `--config XXX.json` 读取，命令行只保留操作名。
未填字段使用默认值；**实际生效的配置会写回 run 目录的 `config.json`**，复现时以该文件为准。

JSON 不允许注释、重复字段、`NaN`/`Infinity`、未知字段或错误类型。

---

## 1. 顶层结构

| 段 | 作用 |
|---|---|
| `model` | 主干路径、精度、注意力后端、VAE 精度 |
| `lora` | LoRA 秩、alpha、注入位置 |
| `sampling` | rollout 的分辨率、步数、CFG、时间表、精度模式 |
| `art` | 优势估计与参考项 |
| `optimizer` | 优化器超参 |
| `training` | 预算、batch 组织、检查点策略 |
| `reward` | 奖励服务的调用方式与版本 |
| `data` | 训练/评估 JSONL、分档、缓存 |
| `logging` | TensorBoard、rollout 图开关 |

---

## 2. 关键字段

### 2.1 `model`

| 字段 | 典型值 | 说明 |
|---|---|---|
| `path` | `$MODEL_ROOT/MOSS-Image-Base` | 权重目录 |
| `dtype` | `bfloat16` | 主干计算精度 |
| `attention_backend` | `cudnn` | 注意力实现 |
| `activation_storage` | `gpu` | 激活驻留位置 |
| `vae_precision` | `float16` | VAE 精度 |
| `vae_conv_backend` | `cudnn_bounded` | VAE 卷积后端 |
| `mxfp8_weight_mode` | `offline` | 预量化权重加载方式 |

### 2.2 `sampling`

| 字段 | 典型值 | 说明 |
|---|---|---|
| `rollout_mode` | **`mxfp8_calibrated`** | 训练允许的唯一 FP8 模式 |
| `height` / `width` | 1024 / 768 | 分辨率 |
| `steps` | 48 | rollout 步数 |
| `group_size` (K) | 8 / 16 | 每 prompt 采样数 |
| `guidance_scale` | 3.0 | CFG 强度 |
| `skip_layers` | `[22]` | 负分支跳过的层 |
| `train_timesteps` | `[0,15,31,47]` | 训练时间点 |
| `shift` | 5.0 | 时间表偏移 |

### 2.3 `art`

| 字段 | 典型值 | 说明 |
|---|---|---|
| `noise_level` | 0.8 | 加噪强度 |
| `feedback` | `scalar` | 反馈类型（标量优势） |
| `reference_mse_beta` | 0.01 / 0.0 | 参考项系数（0 表示不约束偏离 Base） |
| `kl_threshold` / `eps` | 1e-5 | 数值稳定项 |

> `reference_mse_beta=0` 时模型可以自由偏离 Base，配合 rank-sqrt 优势收敛更快，
> 但长期训练存在审美坍缩风险；本实验的 200 更新预算本就为检验这一点而设。

### 2.4 `training`

| 字段 | 典型值 | 说明 |
|---|---|---|
| `max_updates` | 20 / 200 | 优化器更新总预算 |
| `prompts_per_update` (P) | 24 / 8 | 每 collection 的 prompt 数 |
| `rollouts_per_prompt` (K) | 8 / 16 | 同 `sampling.group_size` |
| `training_points_per_trajectory` (T) | 4 | 每条轨迹的训练点 |
| `updates_per_rollout` | 2 | 每次采样做几次更新 |
| `checkpoint_every` | 2 | 检查点间隔 |
| `keep_checkpoints` | 2 | 保留个数（轮转） |
| `max_no_signal_rollouts` | 20 | 无信号保护 |

### 2.5 `reward`

```json
"reward": {
  "kind": "command",
  "command": ["$CONDA_PREFIX/envs/torch_env/bin/python",
              "$MOSS_ROOT/../moss-reward-pipeline/checkpoint_eval/run_reward_with_validation.py",
              "--config", "$MOSS_ROOT/runs/<run>/reward_pipeline.config.json"],
  "timeout": 7200
}
```

---

## 3. 四组典型组合

| 用途 | 分辨率 | P×K×T | 预算 | 单次更新 |
|---|---|---|---|---|
| 小成本 pilot | 768 | 24×8×4 | 20 | ~10 min |
| 正式实验 | 1024 | 24×8×4 | 200 | ~15 min |
| 高采样变体 | 1024 | 24×16×4 | 20 | ~21 min |
| 少卡试跑 | 768 | 8×16×4 | 20 | ~20 min |

## 4. 校验

```bash
# 打印生效配置（含默认值填充）
python scripts/effective_config.py --config configs/my_run.json

# 关键断言：训练模式必须是校准 FP8
python - <<'PY'
import json,sys
c=json.load(open("configs/my_run.json"))
m=c["sampling"]["rollout_mode"]
assert m=="mxfp8_calibrated", f"rollout_mode={m} 不合规"
print("rollout_mode OK:", m)
PY
```
