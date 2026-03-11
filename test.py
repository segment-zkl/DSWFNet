import os
import re
from PIL import Image
import torch
import numpy as np
import pandas as pd
import cv2
import warnings
import torchvision.transforms as transforms
from pathlib import Path
from model.DSWFNet import DSWFNet
from Parameter import DEVICE, SET_NAME, sever_root, FILE_NAME
from skimage.morphology import dilation, square

warnings.filterwarnings("ignore")
device = DEVICE

USE_TTA = True

# 多尺度 TTA 实现
def apply_tta_multiscale(model, input_tensor, scales=[0.75, 1.0, 1.25]):
    from torch.nn.functional import interpolate
    def rotate(x, k):
        return torch.rot90(x, k=k, dims=[2, 3])
    def reverse_rotate(x, k):
        return torch.rot90(x, k=4 - k, dims=[2, 3])

    transforms_tta = [
        (lambda x: x, lambda x: x),
        (lambda x: torch.flip(x, dims=[2]), lambda x: torch.flip(x, dims=[2])),
        (lambda x: torch.flip(x, dims=[3]), lambda x: torch.flip(x, dims=[3])),
        (lambda x: torch.flip(x, dims=[2, 3]), lambda x: torch.flip(x, dims=[2, 3])),
        (lambda x: rotate(x, 1), lambda x: reverse_rotate(x, 1)),
        (lambda x: rotate(x, 2), lambda x: reverse_rotate(x, 2)),
        (lambda x: rotate(x, 3), lambda x: reverse_rotate(x, 3)),
        (lambda x: rotate(torch.flip(x, dims=[2]), 1), lambda x: torch.flip(reverse_rotate(x, 1), dims=[2]))
    ]

    preds = []
    with torch.no_grad():
        for scale in scales:
            h, w = input_tensor.shape[2:]
            new_h, new_w = int(h * scale), int(w * scale)
            resized = torch.nn.functional.interpolate(input_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
            for aug_fn, rev_fn in transforms_tta:
                aug_img = aug_fn(resized)
                out = model(aug_img)
                out = torch.sigmoid(out)
                out = rev_fn(out)
                out = torch.nn.functional.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
                preds.append(out)
    return torch.stack(preds).mean(dim=0)

# 指标计算
def calculate_metrics(tp_num, tn_num, fp_num, fn_num, all_num):
    smooth = 1e-6
    precision = tp_num / (tp_num + fp_num + smooth)
    recall = tp_num / (tp_num + fn_num + smooth)
    f1_score = 2 * precision * recall / (precision + recall + smooth)
    accuracy = (tp_num + tn_num) / (all_num + smooth)
    iou_road = tp_num / (tp_num + fp_num + fn_num + smooth)
    iou_background = tn_num / (tn_num + fp_num + fn_num + smooth)
    miou = (iou_road + iou_background) / 2
    rate_pe = ((tp_num + fp_num) * (tp_num + fn_num) + (tn_num + fn_num) * (tn_num + fp_num)) / (all_num * all_num + smooth)
    kappa = (accuracy - rate_pe) / (1 - rate_pe)
    return precision, recall, f1_score, accuracy, iou_road, iou_background, miou, kappa

def get_label_name(image_name):
    base_name, ext = os.path.splitext(image_name)
    if base_name.endswith("_sat"):
        return base_name[:-4] + "_mask.png"
    elif ext == ".tiff":
        return base_name + ".tif"
    return None

def test():
    test_set_name = SET_NAME
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    test_root = project_root / "dataset" / test_set_name / "test"

    save_root = os.path.join(sever_root, f'result/{test_set_name}')
    test_save_path = os.path.join(save_root, 'test_img')
    test_csv_path = os.path.join(save_root, 'csv')
    os.makedirs(test_save_path, exist_ok=True)
    os.makedirs(test_csv_path, exist_ok=True)

    test_csv = pd.DataFrame(columns=['tp_num', 'tn_num', 'fp_num', 'fn_num', 'all_num',
                                     'precision', 'recall', 'f1_score', 'accuracy',
                                     'iou_road', 'iou_background', 'miou', 'kappa'])

    test_result_csv = pd.DataFrame(columns=['type', 'precision', 'recall', 'f1', 'accuracy', 'iou_road', 'miou', 'kappa'])

    model_path = os.path.join(sever_root, 'checkpoint', FILE_NAME, test_set_name, f'model_val.pth')
    model = DSWFNet()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    transform_sat = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    transform_road = transforms.Compose([
        transforms.ToTensor()
    ])

    all_tp = all_tn = all_fp = all_fn = all_num = 0

    image_names = []  # 保存图像名

    for image_name in os.listdir(test_root):
        if not re.match(r"(.+_sat\.jpg)|(.+\.tiff)", image_name):
            continue

        label_name = get_label_name(image_name)
        image_path = os.path.join(test_root, image_name)
        label_path = os.path.join(test_root, label_name)
        base_name, _ = os.path.splitext(image_name)
        image_names.append(base_name)

        image = cv2.imread(image_path)
        label = cv2.imread(label_path, 0)

        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


        input_tensor = transform_sat(image_pil).unsqueeze(0).to(device)
        label_tensor = transform_road(Image.fromarray(label)).unsqueeze(0).to(device)

        label_bin = (label_tensor > 0.5).float().squeeze().cpu().numpy().astype(np.uint8)

        with torch.no_grad():
            if USE_TTA:
                output = apply_tta_multiscale(model, input_tensor)
            else:
                output = model(input_tensor)
                output = torch.sigmoid(output)

            predicted_prob = output.cpu().numpy().squeeze()
            predicted_label = (predicted_prob > 0.35).astype(np.uint8)

        predicted_gray = (predicted_label * 255).astype(np.uint8)
        Image.fromarray(predicted_gray).save(os.path.join(test_save_path, base_name + "_gray.jpg"))

        color_image = np.zeros((predicted_label.shape[0], predicted_label.shape[1], 3), dtype=np.uint8)
        tp_mask = (predicted_label == 1) & (label_bin == 1)
        tn_mask = (predicted_label == 0) & (label_bin == 0)
        fp_mask = (predicted_label == 1) & (label_bin == 0)
        fn_mask = (predicted_label == 0) & (label_bin == 1)

        color_map = {
            "TP": (255, 255, 255),
            "TN": (0, 0, 0),
            "FP": (0, 0, 255),
            "FN": (255, 0, 0)
        }

        color_image[tp_mask] = color_map["TP"]
        color_image[tn_mask] = color_map["TN"]
        color_image[fp_mask] = color_map["FP"]
        color_image[fn_mask] = color_map["FN"]

        Image.fromarray(color_image).save(os.path.join(test_save_path, base_name + "_pred.jpg"))

        tp, tn, fp, fn = np.sum(tp_mask), np.sum(tn_mask), np.sum(fp_mask), np.sum(fn_mask)
        all_px = tp + tn + fp + fn

        all_tp += tp; all_tn += tn; all_fp += fp; all_fn += fn; all_num += all_px

        precision, recall, f1_score, accuracy, iou_road, iou_background, miou, kappa = \
            calculate_metrics(tp, tn, fp, fn, all_px)

        test_csv.loc[len(test_csv)] = [tp, tn, fp, fn, all_px,
                                       precision, recall, f1_score, accuracy,
                                       iou_road, iou_background, miou, kappa]

    assert len(image_names) == len(test_csv)
    test_csv['img_name'] = image_names
    def extract_img_id(name):
        match = re.match(r'^(\d+)_', name)
        return match.group(1) if match else 'unknown'

    test_csv['img_id'] = test_csv['img_name'].apply(extract_img_id)
    cols = ['img_id', 'img_name'] + [col for col in test_csv.columns if col not in ['img_id', 'img_name']]
    test_csv = test_csv[cols]

    test_csv.to_csv(os.path.join(test_csv_path, "test_results.csv"), index=False)

    final_precision, final_recall, final_f1, final_acc, final_iou_road, final_iou_bg, final_miou, final_kappa = \
        calculate_metrics(all_tp, all_tn, all_fp, all_fn, all_num)

    # 构造 DataFrame，包含所有字段
    test_result_csv = pd.DataFrame(columns=[
        'type', 'precision', 'recall', 'f1', 'accuracy',
        'iou_road', 'iou_background', 'miou', 'kappa'
    ])

    # 添加 Final 指标
    test_result_csv.loc[0] = [
        'Final', final_precision, final_recall, final_f1, final_acc,
        final_iou_road, final_iou_bg, final_miou, final_kappa
    ]

    test_result_csv.to_csv(os.path.join(test_csv_path, "final_metrics.csv"), index=False)

    print("\n===== Final 全像素统计指标 =====")
    print("  ".join([
        f"Precision: {final_precision:.4f}", f"Recall: {final_recall:.4f}", f"F1: {final_f1:.4f}",
        f"Accuracy: {final_acc:.4f}", f"IoU Road: {final_iou_road:.4f}", f"IoU BG: {final_iou_bg:.4f}",
        f"mIoU: {final_miou:.4f}", f"Kappa: {final_kappa:.4f}"
    ]))

if __name__ == "__main__":
    test()
