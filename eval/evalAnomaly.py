# Copyright (c) OpenMMLab. All rights reserved.
import os
import glob
import torch
import random
from PIL import Image
import numpy as np
from erfnet import ERFNet
import os.path as osp
from argparse import ArgumentParser
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

seed = 42

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
NUM_CLASSES = 20
# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

input_transform = Compose(
    [
        Resize((512, 1024), Image.BILINEAR),
        ToTensor(),
        # Normalize([.485, .456, .406], [.229, .224, .225]),
    ]
)

target_transform = Compose(
    [
        Resize((512, 1024), Image.NEAREST),
    ]
)

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


def anomaly_score_from_logits(
    logits: torch.Tensor, method: str, temperature: float = 1.0
) -> np.ndarray:
    temperature = max(float(temperature), 1e-8)
    scaled_logits = logits / temperature

    if method == "msp":
        probs = torch.softmax(scaled_logits, dim=1)
        score = 1.0 - probs.max(dim=1).values
        return score.squeeze(0).detach().cpu().numpy()
    if method == "max_logit":
        score = -logits.max(dim=1).values
        return score.squeeze(0).detach().cpu().numpy()
    if method == "max_entropy":
        log_probs = torch.log_softmax(scaled_logits, dim=1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=1)
        return entropy.squeeze(0).detach().cpu().numpy()
    if method == "rba":
        score = -torch.tanh(logits).sum(dim=1)
        return score.squeeze(0).detach().cpu().numpy()
    raise ValueError(f"Unknown method: {method}")


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )  
    parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--method', default="msp", choices=["msp", "max_logit", "max_entropy", "rba"])
    parser.add_argument('--temperature', type=float, default=1.0)
    args = parser.parse_args()
    anomaly_score_list = []
    ood_gts_list = []
    dataset_name = infer_dataset_name(str(args.input[0]))

    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'a')

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    model = ERFNet(NUM_CLASSES)

    use_cuda = (not args.cpu) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        model = torch.nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
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

    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    print ("Model and weights LOADED successfully")
    model.eval()
    
    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        print(path)
        images = input_transform((Image.open(path).convert('RGB'))).unsqueeze(0).float().to(device)
        with torch.no_grad():
            result = model(images)
        anomaly_result = anomaly_score_from_logits(result, args.method, args.temperature)
        pathGT = path.replace("images", "labels_masks")                
        if "RoadObsticle21" in pathGT:
           pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
           pathGT = pathGT.replace("jpg", "png")                
        if "RoadAnomaly" in pathGT:
           pathGT = pathGT.replace("jpg", "png")  

        if not osp.exists(pathGT):
            continue
        mask = Image.open(pathGT)
        mask = target_transform(mask)
        ood_gts = np.array(mask)

        if "RoadAnomaly" in pathGT:
            ood_gts = np.where((ood_gts==2), 1, ood_gts)
        if ("LostAndFound" in pathGT) or ("LostFound" in pathGT) or ("FS_LostFound_full" in pathGT):
            unique_vals = set(np.unique(ood_gts).tolist())
            # Some preprocessed Fishyscapes/Lost&Found masks are already binary:
            # 0=in-distribution, 1=anomaly, 255=ignore. In that case, keep them as-is.
            if not unique_vals.issubset({0, 1, 255}):
                ood_gts = np.where((ood_gts==0), 255, ood_gts)
                ood_gts = np.where((ood_gts==1), 0, ood_gts)
                ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)

        if "Streethazard" in pathGT:
            ood_gts = np.where((ood_gts==14), 255, ood_gts)
            ood_gts = np.where((ood_gts<20), 0, ood_gts)
            ood_gts = np.where((ood_gts==255), 1, ood_gts)

        if 1 not in np.unique(ood_gts):
            continue              
        else:
             ood_gts_list.append(ood_gts)
             anomaly_score_list.append(anomaly_result)
        del result, anomaly_result, ood_gts, mask
        torch.cuda.empty_cache()

    file.write( "\n")

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

    print(f'Dataset: {dataset_name}')
    print(f'Method: {args.method}')
    print(f'Temperature: {args.temperature}')
    print(f'AUPRC score: {prc_auc*100.0}')
    print(f'FPR@TPR95: {fpr*100.0}')

    file.write(
        '    dataset:' + dataset_name +
        '   method:' + str(args.method) +
        '   temperature:' + str(args.temperature) +
        '   AUPRC score:' + str(prc_auc*100.0) +
        '   FPR@TPR95:' + str(fpr*100.0)
    )
    file.close()

if __name__ == '__main__':
    main()
