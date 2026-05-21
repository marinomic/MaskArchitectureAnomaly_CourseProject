import glob
import os
import os.path as osp
import sys
from argparse import ArgumentParser
from typing import Tuple

_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
sys.path.insert(0, osp.join(_ROOT, "eomt"))
sys.path.insert(0, _ROOT)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from eomt.models.eomt import EoMT
from eomt.models.vit import ViT
from eomt.training.mask_classification_semantic import MaskClassificationSemantic


def average_precision_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float64)
    if y_true.ndim != 1 or y_score.ndim != 1 or y_true.shape[0] != y_score.shape[0]:
        raise ValueError("y_true and y_score must be 1D arrays of the same length")

    pos = int(np.sum(y_true == 1))
    if pos == 0:
        return 0.0

    order = np.argsort(-y_score, kind="mergesort")
    y_true_sorted = y_true[order]

    tp = np.cumsum(y_true_sorted == 1)
    fp = np.cumsum(y_true_sorted == 0)

    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / pos

    distinct_mask = np.r_[True, y_score[order][1:] != y_score[order][:-1]]
    precision = precision[distinct_mask]
    recall = recall[distinct_mask]

    recall = np.r_[0.0, recall]
    precision = np.r_[precision[0], precision]

    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def fpr_at_95_tpr(y_score: np.ndarray, y_true: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float64)
    if y_true.ndim != 1 or y_score.ndim != 1 or y_true.shape[0] != y_score.shape[0]:
        raise ValueError("y_true and y_score must be 1D arrays of the same length")

    pos = int(np.sum(y_true == 1))
    neg = int(np.sum(y_true == 0))
    if pos == 0 or neg == 0:
        return 0.0

    order = np.argsort(-y_score, kind="mergesort")
    y_true_sorted = y_true[order]

    tp = np.cumsum(y_true_sorted == 1)
    fp = np.cumsum(y_true_sorted == 0)

    tpr = tp / pos
    fpr = fp / neg

    idx = np.where(tpr >= 0.95)[0]
    if idx.size == 0:
        return 1.0
    return float(np.min(fpr[idx]))


@torch.no_grad()
def infer_scores_semantic(
    model: MaskClassificationSemantic, img_uint8_chw: torch.Tensor
) -> torch.Tensor:
    imgs = [img_uint8_chw.to(next(model.parameters()).device)]
    img_sizes = [img_uint8_chw.shape[-2:]]

    crops, origins = model.window_imgs_semantic(imgs)
    mask_logits_per_layer, class_logits_per_layer = model(crops)

    mask_logits = mask_logits_per_layer[-1]
    class_logits = class_logits_per_layer[-1]

    mask_logits = F.interpolate(mask_logits, model.img_size, mode="bilinear")
    crop_scores = model.to_per_pixel_logits_semantic(mask_logits, class_logits)
    scores_list = model.revert_window_logits_semantic(crop_scores, origins, img_sizes)
    return scores_list[0]


def anomaly_map_from_scores(scores: torch.Tensor, method: str) -> np.ndarray:
    scores = scores.float()
    raw_scores = scores
    scores = torch.clamp(scores, min=0.0)
    probs = scores / (scores.sum(dim=0, keepdim=True) + 1e-6)

    if method == "msp":
        return (1.0 - probs.max(dim=0).values).detach().cpu().numpy()
    if method == "max_logit":
        logits = torch.log(probs + 1e-12)
        return (-logits.max(dim=0).values).detach().cpu().numpy()
    if method == "max_entropy":
        logits = torch.log(probs + 1e-12)
        entropy = -(probs * logits).sum(dim=0)
        return entropy.detach().cpu().numpy()
    if method == "rba":
        return (-torch.tanh(raw_scores).sum(dim=0)).detach().cpu().numpy()
    raise ValueError(f"Unknown method: {method}")


def load_gt_mask(path: str, pred_hw: Tuple[int, int]) -> np.ndarray:
    pathGT = path.replace("images", "labels_masks")
    if "RoadObsticle21" in pathGT:
        pathGT = pathGT.replace("webp", "png")
    if "fs_static" in pathGT:
        pathGT = pathGT.replace("jpg", "png")
    if "RoadAnomaly" in pathGT:
        pathGT = pathGT.replace("jpg", "png")

    if not osp.exists(pathGT):
        return None

    mask = Image.open(pathGT)
    if mask.size != (pred_hw[1], pred_hw[0]):
        mask = mask.resize((pred_hw[1], pred_hw[0]), Image.NEAREST)
    ood_gts = np.array(mask)

    if "RoadAnomaly" in pathGT:
        ood_gts = np.where((ood_gts == 2), 1, ood_gts)
    if ("LostAndFound" in pathGT) or ("LostFound" in pathGT) or ("FS_LostFound_full" in pathGT):
        unique_vals = set(np.unique(ood_gts).tolist())
        # Some preprocessed Fishyscapes/Lost&Found masks are already binary:
        # 0=in-distribution, 1=anomaly, 255=ignore. In that case, keep them as-is.
        if not unique_vals.issubset({0, 1, 255}):
            ood_gts = np.where((ood_gts == 0), 255, ood_gts)
            ood_gts = np.where((ood_gts == 1), 0, ood_gts)
            ood_gts = np.where((ood_gts > 1) & (ood_gts < 201), 1, ood_gts)

    if "Streethazard" in pathGT:
        ood_gts = np.where((ood_gts == 14), 255, ood_gts)
        ood_gts = np.where((ood_gts < 20), 0, ood_gts)
        ood_gts = np.where((ood_gts == 255), 1, ood_gts)

    return ood_gts


def infer_dataset_name(input_pattern: str) -> str:
    norm = input_pattern.replace("\\", "/")
    parts = [p for p in norm.split("/") if p]
    if "Validation_Dataset" in parts:
        idx = parts.index("Validation_Dataset")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    if len(parts) >= 2:
        return parts[-2]
    return input_pattern


def build_model(args) -> MaskClassificationSemantic:
    img_size = (args.img_size_h, args.img_size_w)
    # Pass ckpt_path so ViT skips downloading external pretrained backbone weights.
    # The full EoMT checkpoint is loaded below and provides the actual parameters.
    encoder = ViT(
        img_size=img_size, backbone_name=args.backbone_name, ckpt_path=args.ckpt
    )
    masked_attn_enabled = not args.masked_attn_disabled
    network = EoMT(
        encoder=encoder,
        num_classes=args.num_classes,
        num_q=args.num_q,
        num_blocks=args.num_blocks,
        masked_attn_enabled=masked_attn_enabled,
    )
    model = MaskClassificationSemantic(
        network=network,
        img_size=img_size,
        num_classes=args.num_classes,
        attn_mask_annealing_enabled=args.attn_mask_annealing_enabled,
        ckpt_path=None,
        load_ckpt_class_head=True,
    )
    use_cuda = (not args.cpu) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device).eval()

    ckpt = model._load_ckpt(args.ckpt, load_ckpt_class_head=True)
    pos_key = "network.encoder.backbone.pos_embed"
    if pos_key in ckpt:
        target_grid = model.network.encoder.backbone.patch_embed.grid_size
        target_n = int(target_grid[0] * target_grid[1])
        if ckpt[pos_key].shape[1] != target_n:
            pos = ckpt[pos_key]
            old_n = int(pos.shape[1])
            old_h = int(round(old_n**0.5))
            old_w = old_h
            if old_h * old_w != old_n:
                raise ValueError(f"Unexpected pos_embed length {old_n} (not square)")

            pos = pos.reshape(1, old_h, old_w, -1).permute(0, 3, 1, 2)
            pos = F.interpolate(
                pos, size=target_grid, mode="bicubic", align_corners=False
            )
            pos = pos.permute(0, 2, 3, 1).reshape(1, target_n, -1)
            ckpt[pos_key] = pos

    model.load_state_dict(ckpt, strict=False)
    return model


def main():
    parser = ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--method", default="msp", choices=["msp", "max_logit", "max_entropy", "rba"])
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--img_size_h", type=int, default=1024)
    parser.add_argument("--img_size_w", type=int, default=1024)
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--num_q", type=int, default=100)
    parser.add_argument("--num_blocks", type=int, default=3)
    parser.add_argument("--backbone_name", default="vit_base_patch14_reg4_dinov2")
    parser.add_argument("--masked_attn_disabled", action="store_true")
    parser.add_argument("--attn_mask_annealing_enabled", action="store_true")
    args = parser.parse_args()

    model = build_model(args)
    dataset_name = infer_dataset_name(str(args.input[0]))

    anomaly_score_list = []
    ood_gts_list = []

    out_path = "results_eomt.txt"
    if not os.path.exists(out_path):
        open(out_path, "w").close()
    f = open(out_path, "a")

    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        img = Image.open(path).convert("RGB")
        img_np = np.array(img)
        img_uint8 = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()

        scores = infer_scores_semantic(model, img_uint8)
        anomaly_map = anomaly_map_from_scores(scores, args.method)

        ood_gts = load_gt_mask(path, pred_hw=anomaly_map.shape)
        if ood_gts is None:
            continue

        if 1 not in np.unique(ood_gts):
            continue

        ood_gts_list.append(ood_gts)
        anomaly_score_list.append(anomaly_map)

    f.write("\n")

    if len(ood_gts_list) == 0:
        print(f"Dataset: {dataset_name}")
        print("No valid samples found (missing labels or no anomaly pixels).")
        f.write(
            "    dataset:"
            + dataset_name
            + "   method:"
            + str(args.method)
            + "   no_valid_samples\n"
        )
        f.close()
        return

    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)

    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)

    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]

    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))

    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))

    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)

    print(f"Model: EoMT")
    print(f"Dataset: {dataset_name}")
    print(f"Method: {args.method}")
    print(f"AUPRC score: {prc_auc*100.0}")
    print(f"FPR@TPR95: {fpr*100.0}")

    f.write(
        "    dataset:"
        + dataset_name
        + "   method:"
        + str(args.method)
        + "   AUPRC score:"
        + str(prc_auc * 100.0)
        + "   FPR@TPR95:"
        + str(fpr * 100.0)
    )
    f.close()


if __name__ == "__main__":
    main()
