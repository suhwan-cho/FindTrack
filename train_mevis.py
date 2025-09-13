from models.evf_sam import EvfSamModel
from datasets.mevis import MeViSDataset
from datasets.transforms_image import ResizeLongestSide
from transformers import AutoTokenizer, BitsAndBytesConfig
from einops import rearrange, repeat
from tqdm import tqdm
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
import deepspeed
import warnings
warnings.filterwarnings('ignore')

def init_models():
    tokenizer = AutoTokenizer.from_pretrained('YxZhang/evf-sam-multitask', padding_side='right', use_fast=False)
    evfsam = EvfSamModel.from_pretrained('YxZhang/evf-sam-multitask', low_cpu_mem_usage=True, cache_dir='../huggingface')
    evfsam = evfsam.cuda()
    evfsam.train()
    return tokenizer, evfsam

def train():

    # initialize distributed training
    if torch.cuda.device_count() > 1:
        deepspeed.init_distributed()
    ds_config = {
        'train_batch_size': 10,
        'train_micro_batch_size_per_gpu': 1,
        'gradient_accumulation_steps': 5,
        'fp16': {
            'enabled': True,
            'initial_scale_power': 8
        },
        'zero_optimization': {
            'stage': 2
        }
    }   

    # initialize models
    tokenizer, model = init_models()
    
    # load dataset
    root = '../DB/RVOS/MeViS'
    img_folder = os.path.join(root, 'train')
    ann_file = os.path.join(root, 'train', 'meta_expressions.json')
    dataset = MeViSDataset(img_folder=img_folder, ann_file=ann_file, tf=ResizeLongestSide(1024),
                           return_masks=True, num_frames=5, max_skip=3)
    
    # setup distributed training
    if torch.cuda.device_count() > 1:
        rank = deepspeed.comm.get_rank()
        world_size = deepspeed.comm.get_world_size()
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    else:
        rank = 0
        sampler = None
    dataset_loader = DataLoader(dataset, batch_size=1, num_workers=4, sampler=sampler)
    optimizer = optim.AdamW(model.parameters(), lr=1e-6)

    # initialize deepspeed
    model, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        model_parameters=model.parameters(),
        config=ds_config
    )
    
    # training loop
    max_epochs = 10
    for epoch in range(1, max_epochs + 1):
        sampler.set_epoch(epoch)
        loss_sum = 0
        bce_loss_sum = 0
        dice_loss_sum = 0
        for iter, data in tqdm(enumerate(dataset_loader, 1), total=len(dataset_loader), desc=f'Epoch {epoch}/{max_epochs}'):
            imgs_sam, imgs_beit, targets = data
            imgs_sam = imgs_sam[:, 0, :, :, :].unsqueeze(1).cuda()
            imgs_beit = imgs_beit[:, 0, :, :, :].unsqueeze(1).cuda()
            B, T, C, H, W = imgs_sam.shape
            resize = [(H, W)]
            original_size = targets['orig_size']
            
            # text pre-process
            exp = targets['caption'][0]
            input_ids = tokenizer(exp, return_tensors='pt')['input_ids'].cuda()
            attn_masks = torch.zeros_like(input_ids)
            input_ids = input_ids.repeat(T, 1)
            attn_masks = attn_masks.repeat(T, 1)
            size_tensor = targets['orig_size']
            size_tuple = tuple(size_tensor.squeeze().tolist())
            resize = targets['resize'][0]

            # image pre-process
            imgs_sam = rearrange(imgs_sam, 'b t c h w -> (b t) c h w', b=B, t=T)
            imgs_beit = rearrange(imgs_beit, 'b t c h w -> (b t) c h w', b=B, t=T)
            masks = rearrange(targets['masks'][:, 0, :, :].unsqueeze(1), 'b t h w -> (b t) h w', b=B, t=T)
            masks = masks.cuda()

            # loss
            optimizer.zero_grad()
            with torch.amp.autocast(enabled=True, device_type='cuda', dtype=torch.float16):
                loss = model(imgs_sam, imgs_beit, input_ids, torch.ones_like(input_ids), None, masks, size_tuple, resize)
            loss_sum += loss['loss'].item()
            bce_loss_sum += loss['mask_bce_loss'].item()
            dice_loss_sum += loss['mask_dice_loss'].item()

            # print stats
            if rank == 0 and (iter + 1) % 100 == 0:
                avg_loss = loss_sum / (iter + 1)
                avg_loss_bce = bce_loss_sum / (iter + 1)
                avg_loss_dice = dice_loss_sum / (iter + 1)
                print('[it{:04d}] loss: {:.5f}, bce loss: {:.5f}, dice loss: {:.5f}'.format(iter + 1, avg_loss, avg_loss_bce, avg_loss_dice))

            # backward
            model.backward(loss['loss'])
            model.step()

        # refresh stats
        loss_sum = 0
        bce_loss_sum = 0
        dice_loss_sum = 0

        # save model
        if rank == 0:
            os.makedirs('weights', exist_ok=True)
            save_path = os.path.join('weights/mevis_{:04d}.pth'.format(epoch))
            torch.save(model.state_dict(), save_path)

if __name__ == '__main__':
    torch.cuda.set_device(0)
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
        train()
