# Results

This folder collects the project's quantitative results in plain-text format, ready to cite in the report. Each file explains how its numbers were obtained.

**Semantic segmentation (Steps 4-5)**
- `step4_semantic_eval.txt` - mIoU on Cityscapes val of the two provided models (EoMT-Cityscapes vs EoMT-COCO), with the fair-comparison strategy across different label spaces.
- `step5_finetuning.txt` - comparison between EoMT-COCO base, the version fine-tuned on Cityscapes, and the provided Cityscapes checkpoint, including the per-class gain.

**Anomaly segmentation (Steps 7-8)**
- `step7_pixel_baselines.txt` - ERFNet (pixel-based) with MSP, Max Logit and Max Entropy on the five anomaly benchmarks.
- `step8_mask_baselines.txt` - EoMT (mask-based) with MSP, Max Logit, Max Entropy and RbA, for the three EoMT checkpoints (Cityscapes, COCO, fine-tuned).
- `step8_temperature_scaling.txt` - MSP at several temperatures per checkpoint, plus the NLL-fitted best temperature.

A formatted PDF version of the temperature-scaling analysis is available as `temperature_scaling_report.pdf`.

The raw `.json` / `.csv` / `.png` files produced by the notebooks stay in their respective run folders and are the source against which these summaries should be verified.
