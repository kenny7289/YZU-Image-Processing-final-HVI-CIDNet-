import numpy as np
import torch
import gradio as gr
from PIL import Image
from net.CIDNet import CIDNet
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import imquality.brisque as brisque
from loss.niqe_utils import *
import platform
import argparse

opt_parser = argparse.ArgumentParser(description='App')
opt_parser.add_argument('--cpu', action='store_true', help='CPU-Only')
opt = opt_parser.parse_args()

if opt.cpu:
    eval_net = CIDNet().cpu()
else:
    eval_net = CIDNet().cuda()
    
eval_net.trans.gated = True
eval_net.trans.gated2 = True

def process_image(input_img,score,model_path,gamma,alpha_s=1.0,alpha_i=1.0):
    torch.set_grad_enabled(False)
    eval_net.load_state_dict(torch.load(os.path.join(directory,model_path), map_location=lambda storage, loc: storage))
    eval_net.eval()
    
    pil2tensor = transforms.Compose([transforms.ToTensor()])
    input = pil2tensor(input_img)
    factor = 8
    h, w = input.shape[1], input.shape[2]
    H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0
    input = F.pad(input.unsqueeze(0), (0,padw,0,padh), 'reflect')
    with torch.no_grad():
        eval_net.trans.alpha_s = alpha_s
        eval_net.trans.alpha = alpha_i
        if opt.cpu:
            output = eval_net(input**gamma)
        else:
            output = eval_net(input.cuda()**gamma)
            
    if opt.cpu:
        output = torch.clamp(output,0,1)
    else:
        output = torch.clamp(output.cuda(),0,1).cuda()
    output = output[:, :, :h, :w]
    enhanced_img = transforms.ToPILImage()(output.squeeze(0))
    if score == 'Yes':
        im1 = enhanced_img.convert('RGB')
        score_brisque = brisque.score(im1) 
        im1 = np.array(im1)
        score_niqe = calculate_niqe(im1)
        return enhanced_img,score_niqe,score_brisque
    else:
        return enhanced_img,0,0

def find_pth_files(directory):
    pth_files = []
    for root, dirs, files in os.walk(directory):
        if 'train' in root.split(os.sep):
            continue
        for file in files:
            if file.endswith('.pth'):
                pth_files.append(os.path.join(root, file))
    return pth_files

def remove_weights_prefix(paths):
    os_name = platform.system()
    if os_name.lower() == 'windows':
        cleaned_paths = [path.replace('weights\\', '') for path in paths]
    elif os_name.lower() == 'linux':
        cleaned_paths = [path.replace('weights/', '') for path in paths]
        
    return cleaned_paths

directory = "weights"
pth_files = find_pth_files(directory)
pth_files2 = remove_weights_prefix(pth_files)

CLAUDE_NOTICE = (
    "此 Demo 環境使用 Claude Code（Anthropic VS Code 擴充功能）參考官方 repo "
    "Fediory/HVI-CIDNet 所建置與修改，用於個人研究論文實驗。"
)

USAGE_GUIDE = """
### 使用說明
1. 上傳一張低光照（low-light）圖片。
2. 在「Model Weights」選擇權重檔，建議使用 `generalization.pth`（泛化能力最佳）。
3. 可調整 `gamma curve`、`Alpha-s`、`Alpha-i` 滑桿來控制亮度、飽和度與整體增強強度。
4. 「Image Score」選 `Yes` 可額外計算 NIQE / BRISQUE 影像品質指標（耗時較長）。
5. 若無 GPU，啟動時加上 `--cpu` 參數即可在 CPU 上執行：`python app.py --cpu`。
"""

interface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(label="Low-light Image", type="pil"),
        gr.Radio(choices=['Yes','No'],label="Image Score",info="Calculate NIQE and BRISQUE, default is \"No\".",value='No'),
        gr.Radio(choices=pth_files2,label="Model Weights",info="Choose your model. The best models are \"SICE.pth\" and \"generalization.pth\".",value=pth_files2[0] if pth_files2 else None),
        gr.Slider(0.1,5,label="gamma curve",step=0.01,value=1.0, info="Lower is lighter, and best range is [0.5,2.5]."),
        gr.Slider(0,2,label="Alpha-s",step=0.01,value=1.0, info="Higher is more saturated."),
        gr.Slider(0.1,2,label="Alpha-i",step=0.01,value=1.0, info="Higher is lighter.")
    ],
    outputs=[
        gr.Image(label="Result", type="pil"),
        gr.Textbox(label="NIQE",info="Lower is better."),
        gr.Textbox(label="BRISQUE",info="Lower is better.")
    ],
    title=f"HVI-CIDNet (Low-Light Image Enhancement) — {CLAUDE_NOTICE}",
    description=USAGE_GUIDE,
    article=f"---\n{CLAUDE_NOTICE}",
    allow_flagging="never"
)

interface.launch(server_port=7862)
