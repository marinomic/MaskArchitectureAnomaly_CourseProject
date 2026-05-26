import json
import os
import os.path as osp
import random
import sys
from argparse import ArgumentParser

import numpy as np
import torch
from torch import nn, optim
from torchvision.datasets import Cityscapes
from torchvision.transforms import Compose, Resize
from torchvision.transforms import ToTensor
from PIL import Image

_EVAL_DIR = osp.dirname(__file__)
_ROOT = osp.abspath(osp.join(_EVAL_DIR, ".."))
sys.path.insert(0, _EVAL_DIR)
sys.path.insert(0, osp.join(_ROOT, "third_party", "temperature_scaling"))

from dataset import cityscapes
from erfnet import ERFNet
from temperature_scaling import _ECELoss


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CLASSES = 20
IGNORE_INDEX = 19


class LabelIdsToTrainIds:
    def __init__(self, ignore_index: int = IGNORE_INDEX):
        mapping = np.full(256, 255, dtype=np.uint8)
        for cls in Cityscapes.classes:
            if cls.id < 0:
                continue
            train_id = cls.train_id
            if train_id == 255 or cls.ignore_in_eval:
                mapping[cls.id] = 255
            else:
                mapping[cls.id] = train_id
        mapping[255] = 255
        self.mapping = mapping
        self.ignore_index = ignore_index

    def __call__(self, image):
        label_ids = np.array(image, dtype=np.uint8)
        train_ids = self.mapping[label_ids]
        train_ids[train_ids == 255] = self.ignore_index
        return torch.from_numpy(train_ids.astype(np.int64)).unsqueeze(0)

input_transform_cityscapes = Compose(
    [
        Resize(512, Image.BILINEAR),
        ToTensor(),
    ]
)

target_transform_cityscapes = Compose(
    [
        Resize(512, Image.NEAREST),
        LabelIdsToTrainIds(),
    ]
)


class SegmentationTemperatureScaler(nn.Module):
    """
    Segmentation adaptation of the official gpleiss/temperature_scaling code.

    We keep the same optimization idea (single learnable temperature optimized
    on validation NLL), but treat valid semantic segmentation pixels as
    classification samples by flattening [N, C, H, W] -> [N*H*W, C].
    """

    def __init__(self, init_temperature: float = 1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * float(init_temperature))

    def temperature_scale(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp_min(1e-8)

    def set_temperature(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        device: torch.device,
    ) -> dict[str, float]:
        nll_criterion = nn.CrossEntropyLoss().to(device)
        ece_criterion = _ECELoss().to(device)

        logits = logits.to(device)
        labels = labels.to(device)

        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print(
            "Before temperature - NLL: %.6f, ECE: %.6f"
            % (before_temperature_nll, before_temperature_ece)
        )

        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=200)

        def eval_closure():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval_closure)

        after_temperature_nll = nll_criterion(
            self.temperature_scale(logits), labels
        ).item()
        after_temperature_ece = ece_criterion(
            self.temperature_scale(logits), labels
        ).item()
        best_temperature = float(self.temperature.item())
        print("Optimal temperature: %.6f" % best_temperature)
        print(
            "After temperature - NLL: %.6f, ECE: %.6f"
            % (after_temperature_nll, after_temperature_ece)
        )

        return {
            "temperature": best_temperature,
            "before_nll": float(before_temperature_nll),
            "after_nll": float(after_temperature_nll),
            "before_ece": float(before_temperature_ece),
            "after_ece": float(after_temperature_ece),
        }


def load_erfnet(args, device: torch.device) -> nn.Module:
    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print("Loading model: " + modelpath)
    print("Loading weights: " + weightspath)

    model = ERFNet(NUM_CLASSES)
    if device.type == "cuda":
        model = torch.nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    def load_my_state_dict(model, state_dict):
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    print(name, " not loaded")
                    continue
            else:
                own_state[name].copy_(param)
        return model

    model = load_my_state_dict(
        model,
        torch.load(
            weightspath,
            map_location=lambda storage, _: storage,
            weights_only=False,
        ),
    )
    print("Model and weights LOADED successfully")
    model.eval()
    return model


def sample_valid_pixels(
    logits: torch.Tensor, labels: torch.Tensor, max_pixels_per_image: int
) -> tuple[torch.Tensor, torch.Tensor]:
    # logits: [N, C, H, W], labels: [N, 1, H, W]
    logits = logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1])
    labels = labels.squeeze(1).reshape(-1)
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


def collect_validation_logits(args, model: nn.Module, device: torch.device):
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
        batch_size=args.batch_size,
        shuffle=False,
    )

    logits_list = []
    labels_list = []

    with torch.no_grad():
        for step, (images, labels, filename, _) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            sampled_logits, sampled_labels = sample_valid_pixels(
                outputs, labels, max_pixels_per_image=args.max_pixels_per_image
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
    parser.add_argument("--loadDir", default="../trained_models/")
    parser.add_argument("--loadWeights", default="erfnet_pretrained.pth")
    parser.add_argument("--loadModel", default="erfnet.py")
    parser.add_argument("--subset", default="val")
    parser.add_argument("--datadir", required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--init-temperature", type=float, default=1.5)
    parser.add_argument(
        "--max-pixels-per-image",
        type=int,
        default=4096,
        help="Randomly sample at most this many valid pixels per image for fitting T.",
    )
    parser.add_argument(
        "--save",
        default="temperature_erfnet_cityscapes_val.json",
        help="Path to save fitted temperature metadata.",
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda" if (not args.cpu) and torch.cuda.is_available() else "cpu"
    )
    model = load_erfnet(args, device)
    logits, labels = collect_validation_logits(args, model, device)

    scaler = SegmentationTemperatureScaler(init_temperature=args.init_temperature).to(
        device
    )
    stats = scaler.set_temperature(logits, labels, device)

    save_path = args.save
    if not osp.isabs(save_path):
        save_path = osp.join(_EVAL_DIR, save_path)
    os.makedirs(osp.dirname(save_path), exist_ok=True)

    payload = {
        "method": "temperature_scaling",
        "model": "ERFNet",
        "subset": args.subset,
        "datadir": args.datadir,
        "max_pixels_per_image": args.max_pixels_per_image,
        **stats,
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved temperature statistics to: {save_path}")


if __name__ == "__main__":
    main()
