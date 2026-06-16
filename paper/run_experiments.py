"""
Synthesize a small low-light test set from scikit-image's bundled public-domain
sample images, run the reproduced HVI-CIDNet (generalization.pth) on CPU, and
compute NIQE / BRISQUE before vs. after enhancement.

This is used as a supplementary, fully reproducible evaluation for the course
project report, since the original DICM/LIME/MEF/NPE/VV benchmark images are
only distributed via Baidu Pan / OneDrive links that are not scriptable here.
"""
import sys
import os
import csv
sys.argv = ['app.py', '--cpu']
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from skimage import data as skdata
import imquality.brisque as brisque
from loss.niqe_utils import calculate_niqe

from net.CIDNet import CIDNet

OUT_DIR = os.path.join(os.path.dirname(__file__), 'results')
SYN_DIR = os.path.join(os.path.dirname(__file__), 'data_synthetic_lowlight')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SYN_DIR, exist_ok=True)

# 8 standard public-domain RGB sample images bundled with scikit-image
SAMPLES = {
    'astronaut': skdata.astronaut,
    'chelsea': skdata.chelsea,
    'coffee': skdata.coffee,
    'rocket': skdata.rocket,
    'motorcycle_left': lambda: skdata.stereo_motorcycle()[0],
    'cat': skdata.cat,
    'hubble': lambda: skdata.hubble_deep_field()[:, :, :3],
    'retina': skdata.retina,
}

DARKEN_GAMMA = 2.6      # >1 darkens the image
DARKEN_SCALE = 0.22     # multiplicative attenuation
NOISE_STD = 0.01        # small sensor-noise approximation


def to_lowlight(img_uint8):
    img = img_uint8.astype(np.float32) / 255.0
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] == 4:
        img = img[:, :, :3]
    dark = (img ** DARKEN_GAMMA) * DARKEN_SCALE
    dark = dark + np.random.normal(0, NOISE_STD, dark.shape)
    dark = np.clip(dark, 0, 1)
    return (dark * 255).astype(np.uint8)


def enhance(eval_net, pil_img):
    pil2tensor = transforms.Compose([transforms.ToTensor()])
    inp = pil2tensor(pil_img)
    factor = 8
    h, w = inp.shape[1], inp.shape[2]
    H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
    padh, padw = H - h, W - w
    inp = F.pad(inp.unsqueeze(0), (0, padw, 0, padh), 'reflect')
    with torch.no_grad():
        eval_net.trans.alpha_s = 1.0
        eval_net.trans.alpha = 1.0
        out = eval_net(inp ** 1.0)
    out = torch.clamp(out, 0, 1)[:, :, :h, :w]
    return transforms.ToPILImage()(out.squeeze(0))


def score(pil_img):
    im_rgb = pil_img.convert('RGB')
    try:
        b = brisque.score(im_rgb)
    except Exception as e:
        b = float('nan')
    try:
        n = calculate_niqe(np.array(im_rgb))
    except Exception as e:
        n = float('nan')
    return n, b


def main():
    np.random.seed(0)
    eval_net = CIDNet().cpu()
    eval_net.trans.gated = True
    eval_net.trans.gated2 = True
    sd = torch.load(
        os.path.join(os.path.dirname(__file__), '..', 'weights', 'LOLv2_syn', 'generalization.pth'),
        map_location='cpu',
    )
    eval_net.load_state_dict(sd)
    eval_net.eval()

    rows = []
    for name, loader in SAMPLES.items():
        arr = loader()
        low_arr = to_lowlight(arr)
        low_img = Image.fromarray(low_arr)
        low_img.save(os.path.join(SYN_DIR, f'{name}_low.png'))

        enh_img = enhance(eval_net, low_img)
        enh_img.save(os.path.join(OUT_DIR, f'{name}_enhanced.png'))

        niqe_low, brisque_low = score(low_img)
        niqe_enh, brisque_enh = score(enh_img)

        rows.append({
            'image': name,
            'niqe_low': niqe_low, 'niqe_enhanced': niqe_enh,
            'brisque_low': brisque_low, 'brisque_enhanced': brisque_enh,
        })
        print(f'{name:18s} NIQE {niqe_low:6.3f} -> {niqe_enh:6.3f} | '
              f'BRISQUE {brisque_low:7.3f} -> {brisque_enh:7.3f}')

    csv_path = os.path.join(OUT_DIR, 'metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    def avg(key):
        vals = [r[key] for r in rows if not np.isnan(r[key])]
        return sum(vals) / len(vals) if vals else float('nan')

    print('\n--- Averages ---')
    print(f"NIQE:    low={avg('niqe_low'):.3f}  enhanced={avg('niqe_enhanced'):.3f}")
    print(f"BRISQUE: low={avg('brisque_low'):.3f}  enhanced={avg('brisque_enhanced'):.3f}")
    print(f'\nSaved per-image metrics to {csv_path}')


if __name__ == '__main__':
    main()
