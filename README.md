# Mask Architectures for Anomaly Segmentation in Road Scenes

This repository is a fork of the [course starting repository](https://github.com/AlessandroMarinai/MaskArchitectureAnomaly_CourseProject), built for the "Comprehensive Road Scene Understanding for Autonomous Driving" project at Politecnico di Torino. We extended it with a quantitative comparison of the two EoMT checkpoints on Cityscapes, a fine-tuning experiment on the COCO-trained model, and pixel-based and mask-based anomaly segmentation baselines with post-hoc scoring methods and temperature scaling.

## Starting point

The base repository provided the ERFNet model definition and pretrained Cityscapes weights, the full EoMT codebase (configs, model, training modules) with COCO and Cityscapes checkpoints, Cityscapes evaluation utilities, and an initial `evalAnomaly.py` with the MaxLogit baseline already implemented using `sklearn` and `ood_metrics`.

## What this project adds

- **Step 4 — EoMT comparison:** built a hand-crafted COCO→Cityscapes class mapping and a shared evaluation pipeline (640×640, same mIoU code) to compare both models on the full Cityscapes val set. FAIR / COMMON / ORACLE readings of the COCO model account for the label-space mismatch. Results: EoMT-Cityscapes 82.15 %, EoMT-COCO 51.13 % (FAIR) / 63.66 % (COMMON).
- **Step 5 — Fine-tuning:** two-stage AdamW fine-tune with AMP: prediction head only first, then full network with a lower backbone LR (5e-6). Best val mIoU: 69.87 %, saved as `eomt_coco_full_finetuned_cityscapes.bin`.
- **Step 7 — Pixel baselines:** rewrote `evalAnomaly.py` to be sklearn-free (AuPRC and FPR@95TPR reimplemented in NumPy), added MSP (1 − max softmax p) and Max Entropy (−Σ p log p) scoring, a `--method`/`--temperature` CLI, and `fit_temperature_erfnet.py` to find the NLL-optimal T on the Cityscapes val set. Evaluated on five anomaly benchmarks.
- **Step 8 — Mask baselines:** wrote `evalAnomaly_eomt.py` converting mask-architecture output to per-pixel scores via Σ_q sigmoid(mask_q) · softmax(class_q)_c, then applied MSP, Max Logit, Max Entropy and RbA (−Σ tanh(f_c)) across all three checkpoints. `fit_temperature_eomt.py` calibrates T per checkpoint.

## Repository overview

- **eval/** — inherited. We rewrote `evalAnomaly.py` (sklearn-free metrics, new scoring methods), and added `evalAnomaly_eomt.py`, `fit_temperature_erfnet.py`, `fit_temperature_eomt.py`, `results_eomt.txt`, and `temperature_erfnet_cityscapes_val.json`. Other scripts (`dataset.py`, `eval_iou.py`, `eval_cityscapes_*.py`) were updated for compatibility.
- **eomt/** — inherited unchanged in code. We added `checkpoints/` with the three model weights (COCO, Cityscapes, fine-tuned).
- **trained_models/** — inherited; pretrained ERFNet weights.
- **notebooks/** — added by us. Four Colab notebooks: `Step4_EoMT_evaluation.ipynb`, `Step_5_Finetune_EoMT_COCO.ipynb`, `Step7_pixel_baselines.ipynb`, `step8_mask_baselines.ipynb`.
- **results/** — added by us. Plain-text result tables for Steps 4, 5, 7, 8, a temperature-scaling analysis, and a formatted PDF summary.
- **third_party/temperature_scaling/** — added by us; the Guo et al. calibration library used by the ERFNet temperature script.
- **Validation_Dataset/** — added by us; the five anomaly benchmarks (SMIYC RA-21, RO-21, Fishyscapes L&F, Fishyscapes Static, Road Anomaly).

## Main results

Best anomaly detection (Max Logit, AuPRC %):

| Model | RA-21 | RO-21 | FS L&F | FS Static | Road Anomaly |
|---|---|---|---|---|---|
| ERFNet (pixel-based) | 38.31 | 4.62 | 3.29 | 9.49 | 15.58 |
| EoMT-Cityscapes | 65.81 | 89.72 | 17.60 | 62.56 | 66.55 |
| EoMT fine-tuned | 66.18 | 89.35 | 23.98 | 68.29 | 77.81 |

Full per-method, per-checkpoint tables and temperature-scaling results are in `results/`.

## Running the notebooks

All notebooks run on Google Colab with a GPU. Mount your Drive with the weights and Cityscapes archives at the paths defined in the first cell, then run top to bottom. The installation cell in Steps 4 and 5 requires a runtime restart on the first run.

## References

ERFNet, Panoptic Segmentation, MaskFormer, Mask2Former, DINOv2, EoMT (CVPR 2025), RbA, Scaling OoD Detection, SegmentMeIfYouCan, Fishyscapes, Cityscapes, COCO — full citations in the project assignment PDF.
