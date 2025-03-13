from evfsam.segment_anything.utils.transforms import ResizeLongestSide
from evfsam.evf_sam import EvfSamModel
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, BitsAndBytesConfig


def sam_preprocess(x: np.ndarray, pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
                   pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1), img_size=1024):

    # Normalize colors
    x = ResizeLongestSide(img_size).apply_image(x)
    h, w = resize_shape = x.shape[:2]
    x = torch.from_numpy(x).permute(2, 0, 1).contiguous()
    x = (x - pixel_mean) / pixel_std

    # Pad
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x, [resize_shape]


def beit3_preprocess(x: np.ndarray, img_size=224) -> torch.Tensor:
    beit_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    return beit_preprocess(x)


def init_models():
    tokenizer = AutoTokenizer.from_pretrained('YxZhang/evf-sam-multitask', padding_side='right', use_fast=False)
    evfsam = EvfSamModel.from_pretrained('YxZhang/evf-sam-multitask', low_cpu_mem_usage=True, cache_dir='../huggingface')
    evfsam = evfsam.cuda()
    evfsam.eval()
    return tokenizer, evfsam
