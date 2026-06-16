import os
from PIL import Image, ImageDraw, ImageFont

BASE = os.path.dirname(__file__)
SYN_DIR = os.path.join(BASE, 'data_synthetic_lowlight')
OUT_DIR = os.path.join(BASE, 'results')
FIG_DIR = os.path.join(BASE, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

NAMES = ['astronaut', 'cat', 'hubble']
THUMB = 220

pairs = []
for name in NAMES:
    low = Image.open(os.path.join(SYN_DIR, f'{name}_low.png')).convert('RGB')
    enh = Image.open(os.path.join(OUT_DIR, f'{name}_enhanced.png')).convert('RGB')
    low.thumbnail((THUMB, THUMB))
    enh.thumbnail((THUMB, THUMB))
    pairs.append((name, low, enh))

cols = len(pairs)
rows = 2
pad = 10
label_h = 22
cell_w = THUMB
cell_h = THUMB
canvas_w = cols * cell_w + (cols + 1) * pad
canvas_h = rows * (cell_h + label_h) + (rows + 1) * pad
canvas = Image.new('RGB', (canvas_w, canvas_h), 'white')
draw = ImageDraw.Draw(canvas)

for i, (name, low, enh) in enumerate(pairs):
    x = pad + i * (cell_w + pad)
    y0 = pad
    canvas.paste(low, (x, y0 + label_h))
    draw.text((x, y0), f'{name} (low-light input)', fill='black')

    y1 = pad + (cell_h + label_h) + pad
    canvas.paste(enh, (x, y1 + label_h))
    draw.text((x, y1), f'{name} (HVI-CIDNet output)', fill='black')

out_path = os.path.join(FIG_DIR, 'qualitative_comparison.png')
canvas.save(out_path)
print('Saved:', out_path)
