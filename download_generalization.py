import os
import torch
from huggingface_hub import hf_hub_download
import safetensors.torch as sf

repo_id = "Fediory/HVI-CIDNet-Generalization"
model_file = hf_hub_download(repo_id=repo_id, filename="model.safetensors", repo_type="model")
state_dict = sf.load_file(model_file)

out_path = os.path.join("weights", "LOLv2_syn", "generalization.pth")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
torch.save(state_dict, out_path)
print("Saved:", out_path, os.path.getsize(out_path), "bytes")
