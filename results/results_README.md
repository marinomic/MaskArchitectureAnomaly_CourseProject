# Results

This folder collects the project's quantitative results in plain-text format, ready to cite in the report.

- **`step4_semantic_eval.txt`** - mIoU on Cityscapes val of the two provided models (EoMT-Cityscapes vs EoMT-COCO), with the fair-comparison strategy across different label spaces.
- **`step5_finetuning.txt`** - comparison between EoMT-COCO base, the version fine-tuned on Cityscapes, and the provided Cityscapes checkpoint, including the per-class gain.
- **`step8_anomaly_baselines.txt`** - AuPRC / FPR95 tables of the post-hoc methods (MSP, Max Logit, Max Entropy, RbA) and of temperature scaling on the anomaly-segmentation benchmarks.

The raw `.json` / `.csv` / `.png` files produced by the notebooks (per-class scores, mIoU curves, figures) stay in their respective run folders and are the source against which these summaries should be verified.
