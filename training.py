import os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
import timm.optim.optim_factory as optim_factory
import datetime
import time
import argparse
import matplotlib.pyplot as plt
from model import *
from dataset import *
from config import *
from tqdm import tqdm
import math

def adjust_learning_rate(optimizer, epoch, config):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < config.warmup_epochs:
        lr = config.lr * epoch / config.warmup_epochs 
    else:
        lr = config.min_lr + (config.lr - config.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - config.warmup_epochs) / (config.num_epoch - config.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

def save_model(config, epoch, model, optimizer, checkpoint_paths):
    os.makedirs(checkpoint_paths, exist_ok=True)
    to_save = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'config': config,
    }
    torch.save(to_save, os.path.join(checkpoint_paths, 'checkpoint.pth'))

def create_readme(config, path):
    print(config.__dict__)
    with open(os.path.join(path, 'README.md'), 'w+') as f:
        print(config.__dict__, file=f)

def get_args_parser():
    parser = argparse.ArgumentParser('Siamese training', add_help=False)
    
    # Training Parameters
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--num_epoch', type=int)
    parser.add_argument('--batch_size', type=int)

    # Model Parameters
    # parser.add_argument('--mask_ratio', type=float)
    # parser.add_argument('--patch_size', type=int)
    parser.add_argument('--embedding_size', type=int)
    # parser.add_argument('--decoder_embed_dim', type=int)
    # parser.add_argument('--depth', type=int)
    parser.add_argument('--num_res', type=int)
    # parser.add_argument('--decoder_num_heads', type=int)
    # parser.add_argument('--mlp_ratio', type=float)

    # Project setting
    parser.add_argument('--root_path', type=str)
    parser.add_argument('--seed', type=str)
    # parser.add_argument('--roi', type=str)
    # parser.add_argument('--aug_times', type=int)
    # parser.add_argument('--num_sub_limit', type=int)

    # parser.add_argument('--include_hcp', type=bool)
    # parser.add_argument('--include_kam', type=bool)

    # parser.add_argument('--use_nature_img_loss', type=bool)
    # parser.add_argument('--img_recon_weight', type=float)
    
    # distributed training parameters
    parser.add_argument('--local_rank', type=int)
                        
    return parser

def update_config(args, config):
    for attr in config.__dict__:
        if hasattr(args, attr):
            if getattr(args, attr) != None:
                setattr(config, attr, getattr(args, attr))
    return config

def train(config):
    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(config.local_rank) 
        torch.distributed.init_process_group(backend='nccl')
    output_path = os.path.join(config.output_path,  '%s'%(datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")))
    config.output_path = output_path
    # logger = wandb_logger(config) if config.local_rank == 0 else None
    logger = None
    
    if config.local_rank == 0:
        os.makedirs(output_path, exist_ok=True)
        create_readme(config, output_path)
    
    device = torch.device(f'cuda:{config.local_rank}') if torch.cuda.is_available() else torch.device('cpu')
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # create dataset and dataloader
    dataset = image_eeg_dataset(config.eeg_signals_path, config.image_path, neg_per_eeg=config.neg_per_eeg)
   
    print(f'Dataset size: {len(dataset)}\n Time len: {dataset.__len__()}')
    sampler = torch.utils.data.DistributedSampler(dataset, rank=config.local_rank) if torch.cuda.device_count() > 1 else None 

    dataloader_eeg_image = DataLoader(dataset, batch_size=config.batch_size, sampler=sampler, 
                shuffle=(sampler is None), pin_memory=True)

    # create model
    model = siamese_model(config.embedding_size)

    model.to(device)
    model_without_ddp = model
    if torch.cuda.device_count() > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, device_ids=[config.local_rank], output_device=config.local_rank, find_unused_parameters=config.use_nature_img_loss)

    param_groups = optim_factory.param_groups_weight_decay(model, config.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=config.lr, betas=(0.9, 0.95))
    print(optimizer)

    if logger is not None:
        logger.watch_model(model,log='all', log_freq=1000)

    total_loss = []
    start_time = time.time()
    print('Start Training the EEG MAE ... ...')
    accum_iter = config.accum_iter
    for ep in range(1,config.num_epoch+1):
        
        if torch.cuda.device_count() > 1: 
            sampler.set_epoch(ep) # to shuffle the data at every epoch

        model.train(True)
        optimizer.zero_grad()
        for data_iter_step, (data_dict) in tqdm(enumerate(dataloader_eeg_image)):
            if data_iter_step % accum_iter == 0:
                adjust_learning_rate(optimizer, data_iter_step / len(dataloader_eeg_image) + ep, config)
            samples_eeg = data_dict['eeg']
            samples_pos = data_dict['pos_img']
            samples_neg = data_dict['neg_img']
            
            samples_eeg = samples_eeg.to(device)
            samples_pos = samples_pos.to(device)
            samples_neg = samples_neg.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=True):
                loss = model(samples_eeg, samples_pos, samples_neg)
            # loss.backward()
            # norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
            # optimizer.step()

            # loss_value = loss.item()

            # if not math.isfinite(loss_value):
            #     print(f"Loss is {loss_value}, stopping training at step {data_iter_step} epoch {epoch}")
            #     sys.exit(1)

            # loss /= accum_iter
            scaler = torch.cuda.amp.GradScaler()
            scaler.scale(loss.sum()).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad()

            total_loss.append(torch.mean(loss).cpu().detach().numpy())
            if device == torch.device('cuda:0'):
                lr = optimizer.param_groups[0]["lr"]
                if data_iter_step%500 == 0:
                    print('train_loss_step:', np.mean(total_loss), loss.sum().item(), 'lr:', lr)

        # if log_writer is not None:
        #     lr = optimizer.param_groups[0]["lr"]
        #     log_writer.log('train_loss_step', np.mean(total_loss), step=epoch)
        #     log_writer.log('lr', lr, step=epoch)
        #     log_writer.log('cor', np.mean(total_cor), step=epoch)
        #     if start_time is not None:
        #         log_writer.log('time (min)', (time.time() - start_time)/60.0, step=epoch)
        if config.local_rank == 0:        
            print(f'[Epoch {ep}] loss: {np.mean(total_loss)}')

        if (ep % 5 == 0 or ep + 1 == config.num_epoch) and config.local_rank == 0: #and ep != 0
            # save models
        # if True:
            save_model(config, ep, model_without_ddp, optimizer, os.path.join(output_path,'checkpoints'))
            
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    # if logger is not None:
    #     logger.log('max cor', np.max(cor_list), step=config.num_epoch-1)
    #     logger.finish()
    return

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    config = config_siamese()
    config = update_config(args, config)
    train(config)