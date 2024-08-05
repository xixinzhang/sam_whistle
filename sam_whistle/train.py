import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import os
import tyro
from tqdm import tqdm

from sam_whistle.dataset import WhistleDataset
from sam_whistle.model import SAM_whistle
from sam_whistle.config import Args



def run(args: Args):
    # Set seed
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    # Load data
    trainset = WhistleDataset(args, 'train')
    dataloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=os.cpu_count(),)
    testset = WhistleDataset(args, 'test')
    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=os.cpu_count(),)
    # Load model
    model = SAM_whistle(args)
    model.to(args.device)
    optimizer = optim.Adam(model.mask_decoder.parameters(), lr=args.lr)
    # scheduler

    # Train model
    for epoch in tqdm(range(args.epochs)):
        for data in dataloader:
            model.train()
            optimizer.zero_grad()
            # forward
            loss = model(data)
            break
            # backward
            optimizer.step()
    # Save model
    print(args)


if __name__ == "__main__":
    args = tyro.cli(Args)
    run(args)