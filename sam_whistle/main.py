import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import os
import tyro
from tqdm import tqdm
import numpy as np
from datetime import datetime
import json
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from dataclasses import asdict

from sam_whistle.datasets.dataset import WhistleDataset, WhistlePatch, custom_collate_fn
from sam_whistle.model import SAM_whistle, Detection_ResNet_BN2, FCN_Spect, FCN_encoder
from sam_whistle.model.fcn_patch import weights_init_He_normal
from sam_whistle.model.loss import Charbonnier_loss, DiceLoss
from sam_whistle.config import FCNSpectConfig, SAMConfig, DWConfig
from sam_whistle.evaluate.eval_conf import evaluate_sam_prediction, evaluate_deep_prediction, evaluate_fcn_spect_prediction
from sam_whistle.utils.visualize import visualize_array

def run_sam(cfg: SAMConfig):
    # Set seed
    # torch.manual_seed(0)
    # torch.cuda.manual_seed(0)
    # torch.backends.cudnn.deterministic = True
    # Set up logging
    timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    if cfg.exp_name is not None:
        cfg.log_dir = os.path.join(cfg.log_dir, str(timestamp)+"-"+cfg.exp_name)
    else:
        cfg.log_dir = os.path.join(cfg.log_dir, str(timestamp))
    if not os.path.exists(cfg.log_dir):
        os.makedirs(cfg.log_dir)
    with open(os.path.join(cfg.log_dir, 'configs.json'), 'w') as f:
        json.dump(asdict(cfg), f, indent=4)
    writer = SummaryWriter(cfg.log_dir)
    # Load model
    model = SAM_whistle(cfg)
    model.to(cfg.device)
    # optimizer
    if cfg.loss_fn == "mse":
        loss_fn = nn.MSELoss()
    elif cfg.loss_fn == "dice":
        loss_fn = DiceLoss()
    elif cfg.loss_fn == "bce_logits":
        loss_fn = torch.nn.BCEWithLogitsLoss()
    if not cfg.freeze_img_encoder:
        encoder_optimizer = optim.AdamW(model.img_encoder.parameters(), lr=cfg.encoder_lr)
    if not cfg.freeze_mask_decoder:
        decoder_optimizer = optim.AdamW(model.decoder.parameters(), lr=cfg.decoder_lr)
    if not cfg.freeze_prompt_encoder:
        prompt_optimizer = optim.AdamW(model.sam_model.prompt_encoder.parameters(), lr=cfg.prompt_lr)

    # Load data
    print("#"*30 + " Loading data...."+"#"*30)
    trainset = WhistleDataset(cfg, 'train' )
    trainloader = DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True, collate_fn= custom_collate_fn)
    testset = WhistleDataset(cfg, 'test')
    testloader = DataLoader(testset, batch_size= 1, shuffle=False, num_workers=cfg.num_workers, collate_fn= custom_collate_fn)
    print(f"Train set size: {len(trainset)}, Test set size: {len(testset)}")

    min_test_loss = torch.inf
    pbar = tqdm(range(cfg.epochs))
    for epoch in pbar:
        # Train model
        batch_losses = []
        model.train()
        for i, data in enumerate(trainloader):
            spect, gt_mask = data['img'], data['mask']
            spect = spect.to(cfg.device)
            gt_mask = gt_mask.to(cfg.device)

            pred_mask = model(spect)
            batch_loss = loss_fn(pred_mask, gt_mask)

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.img_encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), max_norm=1.0)
            encoder_optimizer.step()
            decoder_optimizer.step()
            batch_losses.append(batch_loss.item())
            pbar.set_description(f"batch_loss: {batch_loss.item()}")
            writer.add_scalar('Loss/train_batch', batch_loss.item(), epoch*len(trainloader) + i)

        epoch_loss = np.mean(batch_losses)
        pbar.set_description(f"Epoch {epoch} Loss: {epoch_loss}")
        writer.add_scalar('Loss/train_epoch', epoch_loss, epoch)
        
        # Test model and save Model
        test_loss= evaluate_sam_prediction(cfg, False, model, testloader, loss_fn)
        writer.add_scalar('Loss/test', test_loss, epoch)

        if test_loss < min_test_loss:
            print(f"Saving best model with test loss {test_loss} at epoch {epoch}")
            min_test_loss = test_loss
            if not cfg.freeze_img_encoder:
                torch.save(model.img_encoder.state_dict(), os.path.join(cfg.log_dir, 'img_encoder.pth'))
            if not cfg.freeze_mask_decoder:
                torch.save(model.decoder.state_dict(), os.path.join(cfg.log_dir, 'decoder.pth'))
            if not cfg.freeze_prompt_encoder:
                torch.save(model.sam_model.prompt_encoder.state_dict(), os.path.join(cfg.log_dir, 'prompt_encoder.pth'))
        writer.add_scalar('Loss/test_min', min_test_loss, epoch)

def run_deep_whistle(cfg: DWConfig):
    """replicate Pu's work, fcn on balanced patches"""
    timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    if cfg.exp_name is not None:
        cfg.log_dir = os.path.join(cfg.log_dir, str(timestamp)+"-"+cfg.exp_name)
    else:
        cfg.log_dir = os.path.join(cfg.log_dir, str(timestamp))
    if not os.path.exists(cfg.log_dir):
        os.makedirs(cfg.log_dir)
    with open(os.path.join(cfg.log_dir, 'configs.json'), 'w') as f:
        json.dump(asdict(cfg), f, indent=4)
    writer = SummaryWriter(cfg.log_dir)

    model = Detection_ResNet_BN2(cfg.width)
    model.to(cfg.device)
    model.apply(weights_init_He_normal)
    loss_fn = Charbonnier_loss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.adam_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(cfg.scheduler_stepsize), gamma=cfg.scheduler_gamma)

    # Load data
    print("#"*30 + " Loading data...."+"#"*30)
    trainset = WhistlePatch(cfg, 'train')
    trainloader = DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)
    testset = WhistlePatch(cfg, 'test')
    testloader = DataLoader(testset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,)
    print(f"Train set size: {len(trainset)}, Test set size: {len(testset)}")
    
    min_test_loss = torch.inf
    # Train model
    pbar = tqdm(range(cfg.iter_num))
    it = 0
    epoch = 0
    while it < cfg.iter_num_more: 
        batch_losses = []
        model.train()
        for i, data in enumerate(trainloader):
            pbar.update(1)
            it += 1
            img, mask = data['img'], data['mask']
            img = img.to(cfg.device)
            mask = mask.to(cfg.device)
            pred = model(img)
            batch_loss = loss_fn(pred, mask)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            scheduler.step()
            batch_losses.append(batch_loss.item())
            writer.add_scalar('Loss/train_batch', batch_loss.item(), epoch*len(trainloader) + i)
            if it == cfg.iter_num:
                print("Training complete")
                break
        epoch += 1
        epoch_loss = np.mean(batch_losses)
        pbar.set_description(f"Epoch {epoch} Loss: {epoch_loss}")
        writer.add_scalar('Loss/train_epoch', epoch_loss, epoch)
        
        # Test model and save Model
        test_loss= evaluate_deep_prediction(cfg, False, model, testloader, loss_fn)
        writer.add_scalar('Loss/test', test_loss, epoch)

        if test_loss < min_test_loss:
            print(f"Saving best model with test loss {test_loss} at epoch {epoch}")
            min_test_loss = test_loss
            
            if it < cfg.iter_num:
                torch.save(model.state_dict(), os.path.join(cfg.log_dir, 'model_dw.pth'))
            else:
                torch.save(model.state_dict(), os.path.join(cfg.log_dir, 'model_dw_more.pth'))
        writer.add_scalar('Loss/test_min', min_test_loss, epoch)


def run_fcn_spect(cfg: FCNSpectConfig):
    """Apply FCN directly to spectrograms (imbalanced patches)"""

    timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    if cfg.exp_name is not None:
        cfg.log_dir = os.path.join(cfg.log_dir, str(timestamp)+"-"+cfg.exp_name)
    else:
        cfg.log_dir = os.path.join(cfg.log_dir, str(timestamp))
    if not os.path.exists(cfg.log_dir):
        os.makedirs(cfg.log_dir)
    with open(os.path.join(cfg.log_dir, 'configs.json'), 'w') as f:
        json.dump(asdict(cfg), f, indent=4)
    writer = SummaryWriter(cfg.log_dir)

    model = FCN_Spect(cfg)
    model.to(cfg.device)
    model.apply(weights_init_He_normal)
    loss_fn = Charbonnier_loss()

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.adam_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(cfg.scheduler_stepsize), gamma=cfg.scheduler_gamma)

    # Load data
    print("#"*30 + " Loading data...."+"#"*30)
    assert cfg.spect_cfg.block_multi == 1, "Should use block_multi=1"
    trainset = WhistleDataset(cfg, 'train', spect_nchan=1)
    trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=cfg.num_workers, drop_last=False, collate_fn= custom_collate_fn)
    testset = WhistleDataset(cfg, 'test',spect_nchan=1)
    testloader = DataLoader(testset, batch_size= 1, shuffle=False, num_workers=cfg.num_workers, collate_fn= custom_collate_fn)
    print(f"Train set size: {len(trainset)}, Test set size: {len(testset)}")
    
    model.init_patch_ls(trainset[0]['img'].shape[-2:])
    model.order_pick_patch()
    
    pbar = tqdm(range(cfg.iter_num))
    min_test_loss = torch.inf
    it = 0
    epoch = 0
    # Train model
    while it < cfg.iter_num_more: 
        batch_losses = []
        model.train()
        batch_loss = 0
        for i, data in enumerate(trainloader):
            it += model.patch_num
            img, mask = data['img'], data['mask']
            img = img.to(cfg.device)
            mask = mask.to(cfg.device)
            pred = model(img)
            batch_loss += loss_fn(pred, mask) / cfg.batch_size
            if (i+1) % cfg.batch_size == 0:
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                scheduler.step(model.patch_num * cfg.batch_size)
                batch_losses.append(batch_loss.item())
                pbar.set_description(f"batch_loss: {batch_loss.item()}")
                writer.add_scalar('Loss/train_batch', batch_loss.item(), epoch*len(trainloader) + i)
                batch_loss = 0
            if it == cfg.iter_num:
                print("Training complete")
                break

        epoch += 1
        epoch_loss = np.mean(batch_losses)
        pbar.set_description(f"Epoch {epoch} Loss: {epoch_loss}")
        writer.add_scalar('Loss/train_epoch', epoch_loss, epoch)
        
        # Test model and save Model
        test_loss= evaluate_fcn_spect_prediction(cfg, False, model, testloader, loss_fn)
        writer.add_scalar('Loss/test', test_loss, epoch)

        if test_loss < min_test_loss:
            print(f"Saving best model with test loss {test_loss} at epoch {epoch}")
            min_test_loss = test_loss
            
            if it < cfg.iter_num:
                torch.save(model.state_dict(), os.path.join(cfg.log_dir, 'model_fcn_spect.pth'))
            else:
                torch.save(model.state_dict(), os.path.join(cfg.log_dir, 'model_fcn_spect_more.pth'))
        writer.add_scalar('Loss/test_min', min_test_loss, epoch)


def run_fcn_encoder(args: SAMConfig):
    """Apply pretrained FCN as encoder kernel to spectrograms(imbalanced patches), followed by a decoder"""
    timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    if args.exp_name is not None:
        args.log_dir = os.path.join(args.log_dir, str(timestamp)+"-"+args.exp_name)
    else:
        args.log_dir = os.path.join(args.log_dir, str(timestamp))
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    with open(os.path.join(args.log_dir, 'args.txt'), 'w') as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
    writer = SummaryWriter(args.log_dir)

    model = FCN_encoder(args)
    model.to(args.device)
    loss_fn = DiceLoss()

    if not args.freeze_img_encoder:
        encoder_optimizer = optim.AdamW(model.img_encoder.parameters(), lr=args.fcn_encoder_lr)
    if not args.freeze_mask_decoder:
        decoder_optimizer = optim.AdamW(list(model.decoder.parameters()) + list(model.downsample.parameters()), lr=args.fcn_decoder_lr)


    # Load data
    print("#"*30 + " Loading data...."+"#"*30)
    trainset = WhistleDataset(args, 'train', spect_nchan=1)
    trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=args.num_workers, drop_last=False)
    testset = WhistleDataset(args, 'test',spect_nchan=1)
    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=args.num_workers,)
    print(f"Train set size: {len(trainset)}, Test set size: {len(testset)}")
    model.init_patch_ls()
    
    losses = []
    min_test_loss = torch.inf
    # Train model
    pbar = tqdm(range(args.fcn_encoder_epochs))
    for epoch in pbar:
        epoch_losses = []
        model.train()
        batch_loss = 0
        for i, data in enumerate(trainloader):
            spect, mask = data
            spect = spect.to(args.device)
            mask = mask.to(args.device)
            pred = model(spect)
            batch_loss += loss_fn(pred, mask) / args.spect_batch_size
            if (i+1) % args.spect_batch_size == 0:
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.img_encoder.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), max_norm=1.0)
                encoder_optimizer.step()
                decoder_optimizer.step()
                epoch_losses.append(batch_loss.item())
                pbar.set_description(f"batch_loss: {batch_loss.item()}")
                writer.add_scalar('Loss/train', batch_loss.item(), epoch*len(trainloader) + i)
                batch_loss = 0


        losses.append(epoch_losses)
        pbar.set_description(f"Epoch {epoch} Loss: {np.mean(epoch_losses)}")

        # Test model and save Model
        # model.eval() # not needed as batch size is 1, batch norm is unstable
        test_losses = []
        for i, data in enumerate(testloader):
            with torch.no_grad():
                spect, mask = data
                spect = spect.to(args.device)
                mask = mask.to(args.device)
                pred = model(spect)
                test_loss = loss_fn(pred, mask)
                test_losses.append(test_loss.item())
        test_loss = np.mean(test_losses)
        writer.add_scalar('Loss/test', test_loss, epoch)
        # grid = vutils.make_grid(torch.cat((pred[:8], mask[:8]), dim=0), normalize=True, nrow=4, padding=2)
        # writer.add_image('Mask', grid, epoch)
        print(f"Test Loss: {test_loss}")
        # test_loss = np.mean(epoch_losses)
        if test_loss < min_test_loss:
            print(f"Saving best model with test loss {test_loss} at epoch {epoch}")
            min_test_loss = test_loss
            torch.save(model.state_dict(), os.path.join(args.log_dir, 'model.pth'))



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='sam_whistle', help='Model to run')
    args, remaining = parser.parse_known_args()
    if args.model == 'sam':
        cfg = tyro.cli(SAMConfig, args=remaining)
        run_sam(cfg)
    elif args.model == 'deep':
        cfg = tyro.cli(DWConfig, args=remaining)
        run_deep_whistle(cfg)
    elif args.model == 'fcn_spect':
        cfg = tyro.cli(FCNSpectConfig, args=remaining)
        run_fcn_spect(cfg)
    # elif args.model == 'fcn_encoder':
    #     run_fcn_encoder(cfg)
    else:
        raise ValueError("Model not recognized")