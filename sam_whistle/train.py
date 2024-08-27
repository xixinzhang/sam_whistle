import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import os
import tyro
from tqdm import tqdm
import numpy as np
from datetime import datetime

from sam_whistle.dataset import WhistleDataset
from sam_whistle.model import SAM_whistle
from sam_whistle.config import Args
from sam_whistle import utils



def run(args: Args):
    # Set seed
    # torch.manual_seed(0)
    # torch.cuda.manual_seed(0)
    # torch.backends.cudnn.deterministic = True
    # Load model

    timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    args.save_path = os.path.join(args.save_path, str(timestamp))
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    model = SAM_whistle(args)
    model.to(args.device)
    optimizer = optim.AdamW(model.sam_model.mask_decoder.parameters(), lr=args.lr)
    # scheduler
    # Load data
    trainset = WhistleDataset(args, 'train', model.sam_model.image_encoder.img_size)
    trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=os.cpu_count(), drop_last=True)
    testset = WhistleDataset(args, 'test',model.sam_model.image_encoder.img_size)
    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=os.cpu_count(),)
    
    losses = []
    min_test_loss = torch.inf
    # Train model
    pbar = tqdm(range(args.epochs))
    for epoch in pbar:
        epoch_losses = []
        model.train()
        loss = 0
        for i, data in enumerate(trainloader):
            l, _, _ = model(data)
            loss += l/ args.batch_size
            if (i+1) % args.batch_size == 0:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.sam_model.mask_decoder.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_losses.append(loss.item())
                pbar.set_description(f"batch_loss: {loss.item()}")
                loss = 0
        losses.append(epoch_losses)
        print(f"Epoch {epoch} Loss: {np.mean(epoch_losses)}")
        # # Test model and save Model
        # model.eval()
        # test_losses = []
        # for data in testloader:
        #     with torch.no_grad():
        #         loss, pred_mask = model(data)
        #         test_losses.append(loss.item())
        # test_loss = np.mean(test_losses)
        # print(f"Test Loss: {test_loss}")
        test_loss = np.mean(epoch_losses)
        if test_loss < min_test_loss:
            print(f"Saving best model with test loss {test_loss} at epoch {epoch}")
            min_test_loss = test_loss
            torch.save(model.sam_model.mask_decoder.state_dict(), os.path.join(args.save_path, 'decoder.pth'))
    # Save losses to file
    with open(os.path.join(args.save_path, 'losses.txt'), 'w') as f:
        for epoch, epoch_losses in enumerate(losses):
            f.write(f"Epoch {epoch} Loss: {np.mean(epoch_losses)}\n")

@torch.no_grad()
def evaluate(args: Args):

    print("------------Evaluating model------------")
    model = SAM_whistle(args,)
    model.to(args.device)
    model.sam_model.mask_decoder.load_state_dict(torch.load(os.path.join(args.save_path, 'decoder.pth')))
    output_path = os.path.join(args.save_path, 'predictions')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    trainset = WhistleDataset(args, 'train', model.sam_model.image_encoder.img_size)
    trainloader = DataLoader(trainset, batch_size=1, shuffle=False, num_workers=os.cpu_count(), drop_last=True)
    testset = WhistleDataset(args, 'test',model.sam_model.image_encoder.img_size)
    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=os.cpu_count(),)
    print(f"Train set size: {len(trainset)}, Test set size: {len(testset)}")

    model.eval()
    test_losses = []
    for i, data in enumerate(trainloader):
        with torch.no_grad():
            spect, _, _, masks, points= data 
            loss, pred_mask, low_mask = model(data)
            # utils.visualize_array(low_mask.cpu().numpy(), output_path, i, 'low_res')
            utils.visualize(spect, masks, points, pred_mask, output_path, i)
            test_losses.append(loss.item())
        if i == 10:
            break
    test_loss = np.mean(test_losses)
    print(f"Test Loss: {test_loss}")

if __name__ == "__main__":
    args = tyro.cli(Args)
    if not args.evaluate:
        run(args)
    else:
        evaluate(args)