# 许可

本仓库包含不同性质的产物，适用不同许可。

## 文档与图表

`README.md`、`INDEX.md`、`docs/`、`report/`、`paper/`、`figures/` 采用
**Creative Commons Attribution 4.0 International（CC BY 4.0）** 许可。

您可以自由复制、分发、改编，但须保留作者署名（soyamilk & Cloveriris）与来源链接。

## 代码

`code/`、`tools/` 采用 **Apache License 2.0**。

```text
Copyright 2026 soyamilk, Cloveriris

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## 不随包分发的资产

| 资产 | 原因 | 获取方式 |
|---|---|---|
| MOSS-Image-Base / SFT 权重 | 体积 | Hugging Face `OpenMOSS-Team/*`（Apache-2.0） |
| 奖励模型权重 | 体积 + 上游许可 | 见 `docs/02-models-and-rewards.md` |
| 训练数据集（图像与提示词） | 第三方版权，不可再分发 | 见 `docs/01-datasets.md` |
| 原始日志包（约 1.8 GB） | 含内部路径 | 可向作者申请 |

## 引用

```bibtex
@techreport{soyamilk2026mossposttrain,
  title  = {Post-Training MOSS-Image: Reward-Driven Online Fine-Tuning
            and Few-Step Distillation},
  author = {soyamilk and Cloveriris},
  year   = {2026},
  month  = {11},
  note   = {Experiment report and reproducibility package}
}
```
