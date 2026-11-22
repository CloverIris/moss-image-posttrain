# PixAI JAX validation

The original 486,346,909-parameter FP32 checkpoint was transferred only after the user explicitly allowed transfer using SSH alias 5070Ti. The LAN alias and previous remote alias were verified against the same machine id. The checkpoint SHA256 matches the local file.

The JAX port retains original parameter names and shape validation. No pretrained weight is committed to Git. Production inference is independent of Torch/Transformers/timm.

Independent full Torch comparison: eight 1536x768 generated anime images, resized/padded by the original preprocessor to1008x1008. FP32 probability maximum absolute difference3.8445e-6; a selected32-tag soft reward maximum difference5.3644e-7; all eight reward ranks match. BF16 probability maximum difference.01812 and selected-tag reward maximum difference.00210; one near-tie ordering changes. This small set is not a general recognition-quality benchmark.

Performance on an RTX5070Ti, no preprocessing, loading or first compile: FP32 native B1 .3135s/image, B4 1.2173s/batch. BF16 cuDNN B1 .08393s/image, B4 .33617s/batch. JAX allocation peak across B1/B4: FP32 3.291GB, BF16 2.008GB, not counting external driver/context allocations.

The public --config CLI loaded actual weights and scored two identical images with identical complete target-score dictionaries. 42 CPU tests passed. Full checkpoint schema398tensors, abstract32-block full-resolution graph, and a concrete5184-token cuDNN global block are covered.

Full details: reports/pixai_tagger_jax_validation_20261021.json. Scripts/raw outputs: experiments/pixai_jax_port_20261021/. Default standalone config: configs/reward/pixai_tagger.json (BF16/cuDNN, selected by the user after validation). ViT attention uses cuDNN; the single-query pool and final sigmoid retain FP32 softmax/scoring arithmetic.
