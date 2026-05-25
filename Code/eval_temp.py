import torch
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import os
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from transformers import SwinModel, BertModel, BertTokenizer

from Code.get_dataset import ChartQAEvalDataset
from eval_collate import collate_fn_eval


def save_overlay(original_img_tensor, saliency_tensor, save_path):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    img = original_img_tensor.cpu() * std + mean
    img = img.permute(1,2,0).numpy()
    img = np.clip(img, 0, 1)

    sal = saliency_tensor.squeeze().cpu().numpy()
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-10)
    heatmap = cm.jet(sal)[:, :, :3]

    heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8))
    heatmap_pil = heatmap_pil.resize((img.shape[1], img.shape[0]), Image.BILINEAR)
    heatmap = np.array(heatmap_pil) / 255.0

    overlay = 0.5 * img + 0.5 * heatmap
    overlay = np.clip(overlay, 0, 1)
    plt.imsave(save_path, overlay)


def evaluation(ckpt: str, device: str, batch_size: int,
               img_dir: str, json_path: str):
    from model_swin import SalFormer

    # 加载 tokenizer 和模型
    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased", cache_dir="/tmp/kwang67_cache")
    llm = BertModel.from_pretrained(
        "bert-base-uncased", cache_dir="/tmp/kwang67_cache")
    vit = SwinModel.from_pretrained(
        "microsoft/swin-tiny-patch4-window7-224", cache_dir="/tmp/kwang67_cache")

    model = SalFormer(vit, llm).to(device)
    checkpoint = torch.load(ckpt, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint: {ckpt}")

    # Dataset & DataLoader
    test_set = ChartQAEvalDataset(
        img_dir=img_dir,
        json_path=json_path,
        tokenizer=tokenizer
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn_eval, num_workers=8
    )

    Path('./eval_resultha').mkdir(parents=True, exist_ok=True)

    for batch_idx, (imgs, q_inputs, imgnames, queries, labels) in enumerate(test_loader):
        imgs = imgs.to(device)
        q_inputs = {k: v.to(device) for k, v in q_inputs.items()}

        with torch.no_grad():
            preds = model(imgs, q_inputs)   # [B, 1, H, W]

        for i in range(preds.shape[0]):
        # 用 batch_idx 和 i 拼出全局序号
            global_idx = batch_idx * batch_size + i
            stem = os.path.splitext(imgnames[i])[0]  # 去掉 .png
            save_path = f"./eval_resultha/{stem}_q{global_idx}.png"
            save_overlay(imgs[i], preds[i], save_path)

        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}/{len(test_loader)} done")

    print("Evaluation complete. Results saved to ./eval_resultha/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",     type=str, default='cuda')
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--ckpt",       type=str,
                        default='./ckpt/model_bert_freeze_10kl_5cc_2nss.tar')
    parser.add_argument("--img_dir",    type=str, default='train/png')
    parser.add_argument("--json_path",  type=str, default='train/train_human.json')
    args = parser.parse_args()

    evaluation(
        ckpt=args.ckpt,
        device=args.device,
        batch_size=args.batch_size,
        img_dir=args.img_dir,
        json_path=args.json_path
    )