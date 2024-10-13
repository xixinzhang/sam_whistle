import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import os
import tyro
from tqdm import tqdm
import numpy as np
from datetime import datetime

from sam_whistle.datasets.dataset import WhistleDataset, WhistlePatch
from sam_whistle.model import SAM_whistle, Detection_ResNet_BN2, FCN_Spect, FCN_encoder
from sam_whistle.model.fcn_patch import weights_init_He_normal
from sam_whistle.model.loss import Charbonnier_loss, DiceLoss
from sam_whistle.config import Args
from sam_whistle.evaluate.evaluate import evaluate_sam

from torch.utils.tensorboard import SummaryWriter
import wandb
import torchvision.utils as vutils

from sam_whistle.visualization import visualize_array

def run_sam(args: Args):
    # Set seed
    # torch.manual_seed(0)
    # torch.cuda.manual_seed(0)
    # torch.backends.cudnn.deterministic = True
    # Load model

    timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    if args.exp_name is not None:
        args.save_path = os.path.join(args.save_path, str(timestamp)+"-"+args.exp_name)
    else:
        args.save_path = os.path.join(args.save_path, str(timestamp))
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    with open(os.path.join(args.save_path, 'args.txt'), 'w') as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
    writer = SummaryWriter(args.save_path)

    model = SAM_whistle(args)
    model.to(args.device)
    # model.sam_model.mask_decoder.apply(weights_init)
    if not args.freeze_img_encoder:
        encoder_optimizer = optim.AdamW(model.img_encoder.parameters(), lr=args.encoder_lr)
    if not args.freeze_mask_decoder:
        decoder_optimizer = optim.AdamW(model.decoder.parameters(), lr=args.decoder_lr)
    if not args.freeze_prompt_encoder:
        prompt_optimizer = optim.AdamW(model.sam_model.prompt_encoder.parameters(), lr=args.prompt_lr)
    # scheduler

    # Load data
    print("#"*30 + " Loading data...."+"#"*30)
    trainset = WhistleDataset(args, 'train', model.sam_model.image_encoder.img_size)
    trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=os.cpu_count(), drop_last=True)
    testset = WhistleDataset(args, 'test',model.sam_model.image_encoder.img_size)
    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=os.cpu_count(),)
    print(f"Train set size: {len(trainset)}, Test set size: {len(testset)}")

    losses = []
    min_test_loss = torch.inf
    # Train model
    pbar = tqdm(range(args.epochs))
    for epoch in pbar:
        epoch_losses = []
        model.train()
        batch_loss = 0
        for i, data in enumerate(trainloader):
            l, _, _, _ = model(data)
            batch_loss += l/ args.spect_batch_size
            if (i+1) % args.spect_batch_size == 0:
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.image_encoder.parameters(), max_norm=1.0)
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
        model.eval()
        test_losses = []
        for data in testloader:
            with torch.no_grad():
                batch_loss, pred_mask,  _, gt_mask = model(data)
                test_losses.append(batch_loss.item())
        test_loss = np.mean(test_losses)
        writer.add_scalar('Loss/test', test_loss, epoch)
        grid = vutils.make_grid(torch.cat([pred_mask, gt_mask], dim=0), nrow=1,  padding=2)
        writer.add_image('Mask/pred_gt', grid, epoch)
        print(f"Test Loss: {test_loss}")
        if test_loss < min_test_loss:
            print(f"Saving best model with test loss {test_loss} at epoch {epoch}")
            min_test_loss = test_loss
            if not args.freeze_img_encoder:
                torch.save(model.img_encoder.state_dict(), os.path.join(args.save_path, 'img_encoder.pth'))
            if not args.freeze_mask_decoder:
                torch.save(model.decoder.state_dict(), os.path.join(args.save_path, 'decoder.pth'))
            if not args.freeze_prompt_encoder:
                torch.save(model.sam_model.prompt_encoder.state_dict(), os.path.join(args.save_path, 'prompt_encoder.pth'))

def run_pu(args: Args):
    """replicate Pu's work, fcn on balanced patches"""
     # Set seed
    # torch.manual_seed(0)
    # torch.cuda.manual_seed(0)
    # torch.backends.cudnn.deterministic = True
    # Load model

    timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    if args.exp_name is not None:
        args.save_path = os.path.join(args.save_path, str(timestamp)+"-"+args.exp_name)
    else:
        args.save_path = os.path.join(args.save_path, str(timestamp))
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    with open(os.path.join(args.save_path, 'args.txt'), 'w') as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
    writer = SummaryWriter(args.save_path)

    model = Detection_ResNet_BN2(args.pu_width)
    model.to(args.device)
    model.apply(weights_init_He_normal)
    loss_fn = Charbonnier_loss()
    optimizer = optim.Adam(model.parameters(), lr=args.pu_lr, weight_decay=args.pu_adam_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(args.pu_scheduler_stepsize), gamma=args.pu_scheduler_gamma)

    # Load data
    print("#"*30 + " Loading data...."+"#"*30)
    trainset = WhistlePatch(args, 'train')
    trainloader = DataLoader(trainset, batch_size=args.pu_batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    testset = WhistlePatch(args, 'test')
    testloader = DataLoader(testset, batch_size=args.pu_batch_size, shuffle=False, num_workers=args.num_workers,)
    print(f"Train set size: {len(trainset)}, Test set size: {len(testset)}")
    
    losses = []
    min_test_loss = torch.inf
    # Train model
    pbar = tqdm(range(args.pu_epochs))
    iter = 0
    for epoch in pbar:
        epoch_losses = []
        model.train()
        for i, data in enumerate(trainloader):
            iter += 1
            img, mask = data
            img = img.to(args.device)
            mask = mask.to(args.device)
            pred = model(img)
            batch_loss = loss_fn(pred, mask)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_losses.append(batch_loss.item())
            # pbar.set_description(f"batch_loss: {batch_loss.item()}")
            writer.add_scalar('Loss/train', batch_loss.item(), epoch*len(trainloader) + i)
            if iter == args.pu_iters:
                print("Reached PU iters")
                break
        print(iter)
        losses.append(epoch_losses)
        pbar.set_description(f"Epoch {epoch} Loss: {np.mean(epoch_losses)}")

        # Test model and save Model
        model.eval()  # used as patches in same spect are batched
        test_losses = []
        for data in testloader:
            with torch.no_grad():
                img, mask, _ = data
                img = img.to(args.device)
                mask = mask.to(args.device)
                pred = model(img)
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
            
            if iter <= args.pu_iters:
                torch.save(model.state_dict(), os.path.join(args.save_path, 'model_pu.pth'))
            else:
                torch.save(model.state_dict(), os.path.join(args.save_path, 'model.pth'))


def run_fcn_spect(args: Args):
    """Apply FCN directly to spectrograms(imbalanced patches)"""

    timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    if args.exp_name is not None:
        args.save_path = os.path.join(args.save_path, str(timestamp)+"-"+args.exp_name)
    else:
        args.save_path = os.path.join(args.save_path, str(timestamp))
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    with open(os.path.join(args.save_path, 'args.txt'), 'w') as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
    writer = SummaryWriter(args.save_path)

    model = FCN_Spect(args)
    model.to(args.device)
    model.apply(weights_init_He_normal)
    loss_fn = Charbonnier_loss()

    optimizer = optim.Adam(model.parameters(), lr=args.pu_lr, weight_decay=args.pu_adam_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(args.pu_scheduler_stepsize), gamma=args.pu_scheduler_gamma)

    # Load data
    print("#"*30 + " Loading data...."+"#"*30)
    trainset = WhistleDataset(args, 'train', spect_nchan=1)
    trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=args.num_workers, drop_last=False)
    testset = WhistleDataset(args, 'test',spect_nchan=1)
    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=args.num_workers,)
    print(f"Train set size: {len(trainset)}, Test set size: {len(testset)}")
    
    losses = []
    min_test_loss = torch.inf
    # Train model
    pbar = tqdm(range(args.fcn_spect_epochs))
    for epoch in pbar:
        epoch_losses = []
        model.train()
        model.init_patch_ls(trainset[0][0].shape[-2:])
        batch_loss = 0
        for i, data in enumerate(trainloader):
            spect, mask = data
            spect = spect.to(args.device)
            mask = mask.to(args.device)
            model.order_pick_patch()
            pred, gt_mask = model(spect, mask)
            batch_loss += loss_fn(pred, gt_mask) / args.spect_batch_size
            if (i+1) % args.spect_batch_size == 0:
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                scheduler.step()
                epoch_losses.append(batch_loss.item())
                pbar.set_description(f"batch_loss: {batch_loss.item()}")
                writer.add_scalar('Loss/train', batch_loss.item(), epoch*len(trainloader) + i)
                batch_loss = 0

        losses.append(epoch_losses)
        pbar.set_description(f"Epoch {epoch} Loss: {np.mean(epoch_losses)}")

        # Test model and save Model
        model.eval()
        model.init_patch_ls(testset[0][0].shape[-2:])
        test_losses = []
        for data in testloader:
            with torch.no_grad():
                spect, mask = data
                spect = spect.to(args.device)
                mask = mask.to(args.device)
                model.order_pick_patch()
                pred, gt_mask = model(spect, mask)
                test_loss = loss_fn(pred, gt_mask)
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
            torch.save(model.state_dict(), os.path.join(args.save_path, 'model.pth'))


def run_fcn_encoder(args: Args):
    """Apply pretrained FCN as encoder kernel to spectrograms(imbalanced patches), followed by a decoder"""
    timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    if args.exp_name is not None:
        args.save_path = os.path.join(args.save_path, str(timestamp)+"-"+args.exp_name)
    else:
        args.save_path = os.path.join(args.save_path, str(timestamp))
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    with open(os.path.join(args.save_path, 'args.txt'), 'w') as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
    writer = SummaryWriter(args.save_path)

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
        for i, data in enumerate(trainloader):
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
            torch.save(model.state_dict(), os.path.join(args.save_path, 'model.pth'))



if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.model == 'pu':
        run_pu(args)
    elif args.model == 'sam':
        run_sam(args)
    elif args.model == 'fcn_spect':
        run_fcn_spect(args)
    elif args.model == 'fcn_encoder':
        run_fcn_encoder(args)
    else:
        raise ValueError("Model not recognized")