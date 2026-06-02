# Mask Architectures for Anomaly Segmentation in Road Scenes

> Course project for *Advanced Machine Learning / Deep Learning* - Politecnico di Torino
> Topic: **Comprehensive Road Scene Understanding for Autonomous Driving**

This repository collects the work done on **anomaly (out-of-distribution) segmentation** in road scenes. Starting from the study of semantic, instance and panoptic segmentation, it builds up to the evaluation of *post-hoc* methods that detect unknown objects a model has never seen during training.

The core idea is the following. A segmentation network trained on a closed set of classes (for example the 19 of Cityscapes) works well as long as it sees objects it knows, but in a real driving scenario it can face **anomalous** objects - an animal on the road, debris, an unexpected obstacle - that belong to no known class. Flagging these pixels that the model "cannot explain" is critical for safety. The project explores this problem with two families of architectures: a lightweight convolutional network (**ERFNet**) and a modern transformer-based **mask architecture** (**EoMT**, *Your ViT is Secretly an Image Segmentation Model*), comparing several anomaly-scoring methods.

---

## Contents

- [Project goal](#project-goal)
- [Repository structure](#repository-structure)
- [What we did, step by step](#what-we-did-step-by-step)
- [Running the notebooks (Google Colab)](#running-the-notebooks-google-colab)
- [Main results](#main-results)
- [Datasets and weights](#datasets-and-weights)
- [References](#references)

---

## Project goal

The project has two intertwined goals:

1. **Understand and compare** the semantic, instance and panoptic segmentation paradigms, and in particular the shift from *per-pixel* classification to *mask architectures* (MaskFormer to Mask2Former to EoMT with a DINOv2 backbone).
2. **Design, train and evaluate** an anomaly-segmentation pipeline, applying post-hoc methods (MSP, Max Logit, Max Entropy, RbA) and a temperature-scaling baseline, both on a pixel-based model (ERFNet) and on a mask architecture (EoMT).

---

## Repository structure

```
.
├── README.md
├── notebooks/
│   ├── Step4_EoMT_evaluation.ipynb       # EoMT-Cityscapes vs EoMT-COCO comparison (step 4)
│   ├── Step_5_Finetune_EoMT_COCO.ipynb   # fine-tuning the COCO model on Cityscapes (step 5)
│   ├── Step7_pixel_baselines.ipynb        # ERFNet anomaly baselines (step 7)
│   └── step8_mask_baselines.ipynb         # EoMT anomaly baselines + temperature scaling (step 8)
├── eomt/                                 # EoMT library (cloned dependency)
│   ├── checkpoints/
│   │   ├── eomt_cityscapes.bin           # provided Cityscapes checkpoint
│   │   ├── eomt_coco.bin                 # provided COCO checkpoint
│   │   └── eomt_coco_full_finetuned_cityscapes.bin  # our fine-tuned checkpoint
│   ├── configs/                          # YAML configs for EoMT experiments
│   ├── datasets/                         # dataset loaders (Cityscapes, COCO, ADE20k)
│   ├── models/                           # EoMT, ViT and scale_block definitions
│   └── training/                         # Lightning modules and loss functions
├── eval/                                 # anomaly detection evaluation scripts
│   ├── evalAnomaly.py                    # ERFNet: MSP / MaxLogit / MaxEntropy / RbA scoring
│   ├── evalAnomaly_eomt.py               # EoMT: same scoring methods
│   ├── fit_temperature_erfnet.py         # temperature calibration for ERFNet
│   ├── fit_temperature_eomt.py           # temperature calibration for EoMT
│   ├── erfnet.py / erfnet_nobn.py        # ERFNet model definition
│   ├── eval_iou.py                       # ERFNet mIoU on Cityscapes val
│   ├── temperature_erfnet_cityscapes_val.json  # fitted ERFNet temperature (T=1.3082)
│   ├── results.txt                       # ERFNet anomaly detection results log
│   └── results_eomt.txt                  # EoMT anomaly detection results log
├── results/
│   ├── README.md                         # index of the quantitative results
│   ├── step4_semantic_eval.txt           # mIoU of both models on Cityscapes val
│   ├── step5_finetuning.txt              # COCO base / fine-tuned / EoMT-CS comparison
│   ├── step7_pixel_baselines.txt         # ERFNet post-hoc results (MSP / MaxLogit / MaxEntropy)
│   ├── step8_mask_baselines.txt          # EoMT post-hoc results, all three checkpoints
│   ├── step8_temperature_scaling.txt     # MSP at several temperatures per EoMT checkpoint
│   ├── step8_anomaly_baselines.txt       # earlier partial summary (superseded by the two above)
│   └── temperature_scaling_report.pdf    # formatted PDF of the temperature-scaling analysis
├── trained_models/
│   ├── erfnet_pretrained.pth             # pretrained ERFNet weights (Cityscapes)
│   └── erfnet_encoder_pretrained.pth.tar # pretrained ERFNet encoder
├── Validation_Dataset/                   # anomaly detection benchmarks
│   ├── RoadAnomaly/                      # Road Anomaly (60 images)
│   ├── RoadAnomaly21/                    # SMIYC RA-21 subset (10 images)
│   ├── RoadObsticle21/                   # SMIYC RO-21 (30 images)
│   ├── FS_LostFound_full/                # Fishyscapes Lost & Found (100 images)
│   └── fs_static/                        # Fishyscapes Static (30 images)
├── Cityscapes val/                       # Cityscapes gtFine validation split
└── third_party/
    └── temperature_scaling/              # Guo et al. temperature scaling library
```

---

## What we did, step by step

The numbering follows the project assignment.

**Steps 1-3 - Literature study.** Study of ERFNet for real-time semantic segmentation, of the paper that defined panoptic segmentation, and of mask architectures (MaskFormer, Mask2Former) up to EoMT and the use of DINOv2 as a backbone. This part is documented in the report.

**Step 4 - Comparing the two EoMT models.** In `Step4_EoMT_evaluation.ipynb` we load the two provided checkpoints (one trained on Cityscapes for semantic segmentation, one on COCO for panoptic segmentation) and compare them on the Cityscapes validation set. The non-trivial point, also highlighted by the assignment, is that the two models live in **different label spaces**: COCO has 133 panoptic categories, Cityscapes has 19. For a fair comparison we therefore define a **COCO to Cityscapes mapping** and a single evaluation pipeline used for both models, at the same resolution (640x640). We produce both qualitative visualisations (semantic vs. remapped panoptic prediction) and the quantitative mIoU over the whole validation set.

**Step 5 - Fine-tuning the COCO model.** In `Step5_Finetune_EoMT_COCO.ipynb` we fine-tune the COCO model on the Cityscapes training set for semantic segmentation. Following the assignment's hints, we proceed in two stages: first training **only the prediction head** (backbone frozen, the lighter experiment), then unfreezing the whole network for a **full fine-tune** with a lower learning rate on the backbone. We use AMP (mixed precision) to speed up training and log metrics with Weights & Biases. The final comparison puts three versions side by side: the starting COCO model, our fine-tuned one, and the original Cityscapes checkpoint.

**Step 6 - Anomaly segmentation: task and post-hoc methods.** Study of the problem, of the benchmark datasets (SegmentMeIfYouCan, Fishyscapes) and of the main post-hoc methods (RbA, Scaling OoD Detection).

**Step 7 - Pixel-based baseline (ERFNet).** In `Step7_pixel_baselines.ipynb` we run MSP, Max Logit and Max Entropy on ERFNet across all five anomaly validation sets. Forward pass and anomaly-score computation are decoupled: the logits are saved once, then the three methods are applied on them. Temperature scaling is also fitted on the Cityscapes validation set and applied to MSP. Results are in `results/step7_pixel_baselines.txt`.

**Step 8 - Mask-based baseline (EoMT).** In `step8_mask_baselines.ipynb` we evaluate EoMT with the same three methods plus RbA (a method specific to mask architectures, not applicable to ERFNet) on all five anomaly datasets. We run all three checkpoints: COCO, Cityscapes and the fine-tuned one. Temperature scaling is fitted separately for each checkpoint and the effect on MSP is measured at several fixed temperatures. Results are in `results/step8_mask_baselines.txt` and `results/step8_temperature_scaling.txt`; a formatted summary is in `results/temperature_scaling_report.pdf`.

---

## Running the notebooks (Google Colab)

The notebooks are meant to run on Google Colab with a GPU (e.g. T4). In short:

1. Upload the model weights and the Cityscapes archives to Google Drive at the paths listed in the first cell, or add a shortcut to the shared project folder.
2. Run the cells top to bottom. The installation cell requires a **runtime restart** on the first run (this is noted in the cell's caption).
3. For training logging (Step 5) a Weights & Biases account is needed: the API key is requested at runtime and **must not be written into the notebook**.

---

## Main results

**Semantic segmentation — Cityscapes val (500 images, mIoU)**

| Model | Training | mIoU |
|---|---|---|
| EoMT-COCO (base, cross-domain) | COCO panoptic | 51.13% |
| EoMT-COCO fine-tuned (this work) | + Cityscapes semantic | 69.87% |
| EoMT-Cityscapes (provided checkpoint) | Cityscapes semantic | 82.15% |

**Anomaly segmentation — selected results (Max Logit, AuPRC %)**

| Model | SMIYC RA-21 | SMIYC RO-21 | FS L&F | FS Static | Road Anomaly |
|---|---|---|---|---|---|
| ERFNet (pixel-based) | 38.31 | 4.62 | 3.29 | 9.49 | 15.58 |
| EoMT-Cityscapes (mask-based) | 65.81 | 89.72 | 17.60 | 62.56 | 66.55 |
| EoMT fine-tuned (mask-based) | 66.18 | 89.35 | 23.98 | 68.29 | 77.81 |

Full per-method, per-checkpoint tables and temperature-scaling analysis are in the [`results/`](results/) folder.

---

## Datasets and weights

- **Cityscapes** (`leftImg8bit` + `gtFine`): training and validation for semantic segmentation.
- **COCO panoptic**: label space of the pretrained model.
- **Anomaly validation set** (`Anomaly_Validation_Datasets.zip`): SMIYC RA-21, SMIYC RO-21, Fishyscapes L&F, Fishyscapes Static, Road Anomaly.
- **Weights**: `eomt_coco.bin`, `eomt_cityscapes.bin` and the fine-tuned version, provided in the project Drive folder.

---

## References

The bibliographic references (ERFNet, Panoptic Segmentation, MaskFormer, Mask2Former, DINOv2, EoMT, RbA, Scaling OoD, LoRA, SegmentMeIfYouCan, Fishyscapes, Cityscapes, COCO) are listed in the project assignment and in the report.
