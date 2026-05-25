import torch
from torch.utils.data import DataLoader
from env import *

import argparse
from dataset_new import ImagesWithSaliency
from torchvision.utils import save_image
from transformers import SwinModel
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def save_overlay(original_img_tensor, saliency_tensor, save_path):
    """
    original_img_tensor: [3, H, W] normalized tensor
    saliency_tensor:     [1, H, W] 0~1 tensor
    """
    # 反归一化原图
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    img = original_img_tensor.cpu() * std + mean
    img = img.permute(1,2,0).numpy()
    img = np.clip(img, 0, 1)

    # saliency map → colormap
    sal = saliency_tensor.squeeze().cpu().numpy()  # [H, W]
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-10)  # 归一化到0~1
    heatmap = cm.jet(sal)[:, :, :3]  # [H, W, 3], 去掉alpha通道

    # resize heatmap到原图尺寸
    heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8))
    heatmap_pil = heatmap_pil.resize((img.shape[1], img.shape[0]), Image.BILINEAR)
    heatmap = np.array(heatmap_pil) / 255.0

    # overlay
    overlay = 0.5 * img + 0.5 * heatmap
    overlay = np.clip(overlay, 0, 1)

    plt.imsave(save_path, overlay)

def evaluation(Model:str, ckpt: str, device, batch_size:int):
    from model_swin import SalFormer
    from transformers import BertModel
    from tokenizer_bert import padding_fn
    
    if Model == 'bert': # text encoder
        llm = BertModel.from_pretrained("bert-base-uncased", cache_dir="/tmp/kwang67_cache")
        print('BertModel loaded')
    else:
        print('model not available, possiblilities: llama, bloom, bert')
        return

    test_set = ImagesWithSaliency("data/test.npy")

    Path('./eval_result').mkdir(parents=True, exist_ok=True)

    vit = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224", cache_dir="/tmp/kwang67_cache")

    model = SalFormer(vit, llm).to(device)
    checkpoint = torch.load(ckpt)
    model.load_state_dict(checkpoint['model_state_dict']) #load trained weights
    model.eval()

    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=padding_fn, num_workers=8)

    for batch, (img, input_ids, fix, hm, name) in enumerate(test_dataloader): #for each example in the test dataset
        img = img.to(device)
        input_ids = input_ids.to(device)

        with torch.no_grad():
            y = model(img, input_ids) # predicted saliency map

        for i in range(y.shape[0]):
            save_overlay(img[i],                        # 原图tensor
                y[i],                          # saliency map
                f"./eval_result/{name[i]}"    # 保存路径
            )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='bert')
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--ckpt", type=str, default='./ckpt/model_bert_freeze_10kl_5cc_2nss.tar')
    args = vars(parser.parse_args())

    evaluation(Model = args['model'], device = args['device'], ckpt = args['ckpt'], batch_size = args['batch_size'])
