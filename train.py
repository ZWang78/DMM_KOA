import torch
import torch.utils.data as data
import torch.nn as nn
import numpy as np
from torchvision import transforms

from dataset import KneeGradingDataset
from unet import UNet
from transModel import registUnetBlock, gradientLoss, crossCorrelation3D
from diffusion import GaussianDiffusion

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def main():
    # Data transformations
    data_transform = {
        "train": transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ]),
        "val": transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    }

    # Initialize dataset
    train_ds = KneeGradingDataset('./OAI_m', data_transform["train"], data_transform["val"], stage='train')
    train_loader = data.DataLoader(train_ds, batch_size=1, shuffle=True)

    # Initialize model
    model = UNet(
        in_channel=3,         # We will cat [x_start, x_T, x_noisy] -> 3 channels
        out_channel=1,
        inner_channel=8,
        channel_mults=(1, 2, 2, 4),
        attn_res=[10],
        res_blocks=1,
        dropout=0,
        with_time_emb=True,
        image_size=[128, 128]
    ).to(device)

    trans = registUnetBlock(
        input_nc=2,            # [moving_image, code/noise]
        encoder_nc=[16, 32, 32, 32, 32],
        decoder_nc=[32, 32, 32, 8, 8, 2]
    ).to(device)

    # Instantiate losses & diffusion
    loss_ncc = crossCorrelation3D(1, kernel=(9, 9)).to(device)
    loss_reg = gradientLoss("l2").to(device)
    gaussian_diffusion = GaussianDiffusion(
        timesteps=300,
        beta_schedule='linear',
        loss_ncc=loss_ncc,
        loss_reg=loss_reg
    )

    # Optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    optimizer1 = torch.optim.Adam(trans.parameters(), lr=0.0002)

    # Training loop
    epochs = 30000
    for epoch in range(epochs):
        for step, (images0, labels0, images4, labels4) in enumerate(train_loader):
            images0 = images0.to(device)
            images4 = images4.to(device)

            # randomly choose a diffusion timestep
            t = torch.randint(0, 300, (1,), device=device).long()

            # compute loss
            loss, output, x_T, predicted_noise, x_start, flow, x_noisy = \
                gaussian_diffusion.train_losses(model, trans, images0, images4, t)

            # backprop
            optimizer.zero_grad()
            optimizer1.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer1.step()

        # log every 50 epochs
        if epoch % 50 == 0:
            print(f"epoch: {epoch}  Loss: {loss.item()}")
            # Optionally save
            # torch.save(model, f"model_epoch_{epoch}.pth")
            # torch.save(trans, f"trans_epoch_{epoch}.pth")

if __name__ == "__main__":
    main()
