# NOTICE

本产物包含或派生自以下第三方工作。原始版权归各自作者所有。

## 1. MOSS-Image

- **版权**：OpenMOSS Team
- **许可**：Apache License 2.0
- **来源**：<https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base>
- **技术报告**：arXiv:2610.23374
- **说明**：本研究的基座模型与蒸馏教师均基于 MOSS-Image-Base；`code/` 中的
  数据管线与采样器实现派生自其公开代码，并按 Apache-2.0 要求保留本声明。

## 2. BA-solver

- **版权**：HKUST-LongGroup
- **来源**：<https://github.com/HKUST-LongGroup/BA-solver>
- **说明**：少步蒸馏使用的 BA 采样器（双锚点前向 + 后向探测 + Gauss–Lobatto 校正）
  参考其实现，JAX 版本为独立重写并与官方 PyTorch 小张量对拍。

## 3. 审美奖励模型

| 分量 | 说明 |
|---|---|
| Kawai | 动漫插画审美分档分类器（社区公开权重） |
| DeepGHS | 动漫审美百分位回归模型（社区公开权重） |
| HPSv3++ | 人类偏好打分模型，Qwen3-VL-8B 主干（公开权重） |

本研究**不包含**上述权重文件，亦不对其许可作出任何声明。复现者需自行获取并遵守
各上游许可（多为研究用途限制）。具体版本号记录在每个 run 的
`reward_pipeline.config.json` 中。

## 4. 框架与库

| 组件 | 许可 |
|---|---|
| JAX / jaxlib / flax / optax | Apache-2.0 |
| PyTorch | BSD-3-Clause |
| Hugging Face transformers / diffusers / accelerate | Apache-2.0 |
| Pillow | HPND |
| python-docx | MIT |

## 5. 数据集

训练数据为**自建审美数据集**，图像与提示词版权属各自原作者与平台用户，
不随本包分发，也不得二次分发。详见 `docs/01-datasets.md`。

## 6. 免责声明

本产物为研究性质的实验记录。报告中记录的失败、中断与负向结果均如实保留，
不构成对任何模型、方法或产品的性能承诺。使用者应自行评估合规性与适用性。
