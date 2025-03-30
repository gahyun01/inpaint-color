import math
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.optim import Adam
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


class UNetColorization(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super(UNetColorization, self).__init__()

        def down_block(in_feat, out_feat, normalize=True):
            layers = [
                nn.Conv2d(in_feat, out_feat, kernel_size=4, stride=2, padding=1)
            ]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        def up_block(in_feat, out_feat, dropout=0.0):
            layers = [
                nn.ConvTranspose2d(
                    in_feat, out_feat, kernel_size=4, stride=2, padding=1
                ),
                nn.BatchNorm2d(out_feat),
                nn.ReLU(inplace=True),
            ]
            if dropout:
                layers.append(nn.Dropout(dropout))
            return nn.Sequential(*layers)

        self.down1 = down_block(in_channels, 64, normalize=False)
        self.down2 = down_block(64, 128)
        self.down3 = down_block(128, 256)
        self.down4 = down_block(256, 512)
        self.down5 = down_block(512, 512)
        self.down6 = down_block(512, 512)
        self.down7 = down_block(512, 512)
        self.down8 = down_block(512, 512, normalize=False)

        self.up1 = up_block(512, 512, dropout=0.5)
        self.up2 = up_block(1024, 512, dropout=0.5)
        self.up3 = up_block(1024, 512, dropout=0.5)
        self.up4 = up_block(1024, 512)
        self.up5 = up_block(1024, 256)
        self.up6 = up_block(512, 128)
        self.up7 = up_block(256, 64)
        self.up8 = nn.Sequential(
            nn.ConvTranspose2d(
                128, out_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8)
        u2 = self.up2(torch.cat([u1, d7], 1))
        u3 = self.up3(torch.cat([u2, d6], 1))
        u4 = self.up4(torch.cat([u3, d5], 1))
        u5 = self.up5(torch.cat([u4, d4], 1))
        u6 = self.up6(torch.cat([u5, d3], 1))
        u7 = self.up7(torch.cat([u6, d2], 1))
        u8 = self.up8(torch.cat([u7, d1], 1))

        return u8


class ColorizationDataset(Dataset):
    def __init__(self, gray_path, color_path, transform=None):
        self.gray_path = gray_path
        self.color_path = color_path
        self.transform = transform
        self.img_names = os.listdir(gray_path)
        
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        gray_img = Image.open(os.path.join(self.gray_path, img_name)).convert("L")
        color_img = Image.open(os.path.join(self.color_path, img_name)).convert("RGB")
        
        if self.transform:
            gray_img = self.transform(gray_img)
            color_img = self.transform(color_img)
            
        return gray_img, color_img

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())

    model = UNetColorization()
    loss_fn = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=3e-4)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    model.to(device)

    gray_path = "./open/train_mask/"
    color_path = "./open/train_gt/"

    # Initialize dataset and dataloader
    dataset = ColorizationDataset(gray_path, color_path, transform)
    BATCH_SIZE = 16
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )

    min_loss = 1
    for epoch in tqdm(range(1, 10 + 1)):
        for gray, color in tqdm(dataloader, leave=False):
            optimizer.zero_grad()
            
            gray = gray.to(device)
            color = color.to(device)
            
            pred = model(gray)
            loss = loss_fn(pred, color)
            loss.backward()
            optimizer.step()

            if min_loss > loss.item():
                min_loss = loss.item()
                torch.save(model.state_dict(), "./model/colorization.pth")
                # print(f"Model saved with loss: {min_loss}")


# 추론 코드
# pred = pred.squeeze(0).cpu().detach().numpy()  # Remove batch dimension and move to CPU
# pred = (pred * 255).astype(np.uint8)  # Convert to uint8 format

# pred = np.transpose(pred, (1, 2, 0))  # Change from (C, H, W) to (H, W, C)
# pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)  # Convert from RGB to BGR for OpenCV

# cv2.imwrite("./TRAIN_00000_color.png", pred)  # Save
# cv2.imshow("Colorized Image", pred)  # Display
