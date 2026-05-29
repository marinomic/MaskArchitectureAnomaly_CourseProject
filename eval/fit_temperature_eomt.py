import json
import os
import os.path as osp
from argparse import ArgumentParser

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor

from dataset import cityscapes
from evalAnomaly_eomt import build_model, infer_scores_semantic
from fit_temperature_erfnet import IGNORE_INDEX, LabelIdsToTrainIds, SegmentationTemperatureScaler


input_transform_cityscapes = Compose(
    [
        Resize((512, 1024), Image.BILINEAR),
        ToTensor(),
    ]
)

target_transform_cityscapes = Compose(
    [
        Resize((512, 1024), Image.NEAREST),
        LabelIdsToTrainIds(),
    ]
)


def sample_valid_pixels(
    logits: torch.Tensor, labels: torch.Tensor, max_pixels_per_image: int
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = logits.permute(1, 2, 0).reshape(-1, logits.shape[0])
    labels = labels.squeeze(0).reshape(-1)
    valid_mask = labels != IGNORE_INDEX
    logits = logits[valid_mask]
    labels = labels[valid_mask]

    if logits.numel() == 0:
        return logits, labels

    if max_pixels_per_image > 0 and logits.shape[0] > max_pixels_per_image:
        idx = torch.randperm(logits.shape[0])[:max_pixels_per_image]
        logits = logits[idx]
        labels = labels[idx]

    return logits.cpu(), labels.cpu()


def collect_validation_logits(args, model: torch.nn.Module):
    if not osp.exists(args.datadir):
        raise FileNotFoundError(f"datadir does not exist: {args.datadir}")

    loader = torch.utils.data.DataLoader(
        cityscapes(
            args.datadir,
            input_transform_cityscapes,
            target_transform_cityscapes,
            subset=args.subset,
            label_suffix="_labelIds.png",
        ),
        num_workers=args.num_workers,
        batch_size=1,
        shuffle=False,
    )

    logits_list = []
    labels_list = []

    with torch.no_grad():
        for step, (images, labels, filename, _) in enumerate(loader):
            image_uint8 = (images[0] * 255.0).round().clamp(0, 255).to(torch.uint8)
            image_uint8 = image_uint8.to(next(model.parameters()).device)
            scores = infer_scores_semantic(model, image_uint8)

            sampled_logits, sampled_labels = sample_valid_pixels(
                scores, labels[0], args.max_pixels_per_image
            )
            if sampled_logits.numel() == 0:
                continue

            logits_list.append(sampled_logits)
            labels_list.append(sampled_labels)

            if step % 25 == 0:
                print(
                    f"[{step}] collected {sampled_logits.shape[0]} valid pixels "
                    f"from {osp.basename(filename[0])}"
                )

    if len(logits_list) == 0:
        raise RuntimeError("No valid pixels were collected from the validation set.")

    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    print(f"Collected logits: {tuple(logits.shape)}, labels: {tuple(labels.shape)}")
    return logits, labels


def main():
    parser = ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--datadir", required=True)
    parser.add_argument("--subset", default="val")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--init-temperature", type=float, default=1.5)
    parser.add_argument("--max-pixels-per-image", type=int, default=4096)
    parser.add_argument("--save", default="")
    parser.add_argument("--img_size_h", type=int, default=640)
    parser.add_argument("--img_size_w", type=int, default=640)
    parser.add_argument("--backbone_name", default="vit_base_patch14_reg4_dinov2")
    parser.add_argument("--masked_attn_disabled", action="store_true")
    parser.add_argument("--attn_mask_annealing_enabled", action="store_true")
    args = parser.parse_args()

    model = build_model(args)
    logits, labels = collect_validation_logits(args, model)

    device = next(model.parameters()).device
    scaler = SegmentationTemperatureScaler(init_temperature=args.init_temperature).to(
        device
    )
    stats = scaler.set_temperature(logits, labels, device)

    save_path = args.save.strip()
    if not save_path:
        ckpt_name = osp.splitext(osp.basename(args.ckpt))[0]
        save_path = f"temperature_{ckpt_name}_{args.subset}.json"
    if not osp.isabs(save_path):
        save_path = osp.join(osp.dirname(__file__), save_path)
    os.makedirs(osp.dirname(save_path), exist_ok=True)

    payload = {
        "method": "temperature_scaling",
        "model": "EoMT",
        "checkpoint": args.ckpt,
        "subset": args.subset,
        "datadir": args.datadir,
        "max_pixels_per_image": args.max_pixels_per_image,
        "img_size_h": args.img_size_h,
        "img_size_w": args.img_size_w,
        **stats,
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved temperature statistics to: {save_path}")


if __name__ == "__main__":
    main()
