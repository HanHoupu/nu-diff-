import torch
from transformers import AutoModel

def load_backbone(name="bert-base-uncased", train_backbone=False):
    model = AutoModel.from_pretrained(name)
    if not train_backbone:
        for p in model.parameters():
            p.requires_grad_(False)
    return model
