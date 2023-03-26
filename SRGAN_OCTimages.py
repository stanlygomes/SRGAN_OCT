import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from datetime import datetime
from torchvision.utils import save_image
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
from torchvision.models import vgg19
from tqdm import tqdm
from torch.autograd import Variable
from torch.nn import functional as F
import torchvision.transforms.functional as TF

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

# Import dataset and scale images down to 32x32
class RetinaDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['DME', 'DRUSEN']
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.images = []
        self.labels = []

        for cls_name in self.classes:
            cls_path = os.path.join(self.root_dir, 'train', cls_name)
            for img_name in os.listdir(cls_path):
                img_path = os.path.join(cls_path, img_name)
                self.images.append(img_path)
                self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_filename = self.images[index]
        label = self.labels[index]
        # Load the high-resolution image
        hr_img = Image.open(img_filename).convert('RGB')
        hr_transform = transforms.Resize((128, 128))
        hr_img_128 = hr_transform(hr_img)
        
        # Apply a transform to generate the low-resolution image
        lr_transform = transforms.Resize((32, 32))
        lr_img = lr_transform(hr_img)
        
        if self.transform:
            lr_img = self.transform(lr_img)
            hr_img_128 = self.transform(hr_img_128)
            
        return lr_img, hr_img_128
    
# Define the SRGAN generator network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Feature extraction
        self.conv1 = nn.Conv2d(3, 64, 9, 1, 4)
        self.relu1 = nn.PReLU()
        
        # Residual Blocks
        self.resBlock1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64)
        )
        self.resBlock2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64)
        )
        self.resBlock3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64)
        )
        self.resBlock4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64)
        )
        self.resBlock5 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64)
        )

        # Upsampling
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pixshuffle = nn.PixelShuffle(2)
        self.relu2 = nn.PReLU()

        self.conv3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pixshuffle2 = nn.PixelShuffle(2)
        self.relu3 = nn.PReLU()

        # Reconstruction
        self.conv4 = nn.Conv2d(64, 3, 9, 1, 4)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        
        temp = out
        
        res1 = out
        out = self.resBlock1(out)
        out += res1
        
        res2 = out
        out = self.resBlock2(out)
        out += res2
        
        res3 = out
        out = self.resBlock3(out)
        out += res3
        
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.relu3(self.bn3(self.conv3(out)))
        
        # DEBUG
        # print(f'Out shape: {out.shape}')
        # print(f'Residual shape: {residual.shape}')
        out += temp
        
        out = self.conv4(out)

        return out

# Define the SRGAN discriminator network
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super(Discriminator, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, features, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(features, features*2, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(features*2)
        self.conv4 = nn.Conv2d(features*2, features*2, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(features*2, features*4, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(features*4)
        self.conv6 = nn.Conv2d(features*4, features*4, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(features*4, features*8, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(features*8)
        self.conv8 = nn.Conv2d(features*8, features*8, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(features*8*8*8, 1024)
        self.fc2 = nn.Linear(1024, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = self.lrelu(self.conv1(x))
        out = self.lrelu(self.conv2(out))
        out = self.lrelu(self.bn3(self.conv3(out)))
        out = self.lrelu(self.conv4(out))
        out = self.lrelu(self.bn5(self.conv5(out)))
        out = self.lrelu(self.conv6(out))
        out = self.lrelu(self.bn7(self.conv7(out)))
        out = self.lrelu(self.conv8(out))
        out = out.view(out.size(0), -1)
        out = self.lrelu(self.fc1(out))
        out = self.sigmoid(self.fc2(out))
        
        return out
    
# Define Loss Functions
criterion_GAN = nn.BCELoss()
criterion_content = nn.MSELoss()

generator = Generator().to(device)
discriminator = Discriminator().to(device)
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
# # Set hyperparameters
batch_size = 32
num_epochs = 150

SRGAN_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

octDataset = RetinaDataset('Data', transform=SRGAN_transform)
# Split training and validation set
# oct_train, oct_val = random_split(octDataset, [0.7, 0.3])
# trainLoader = DataLoader(oct_train, batch_size=batch_size, shuffle=True)
# valLoader = DataLoader(oct_val, batch_size=1, shuffle=False)
trainLoader = DataLoader(octDataset, batch_size=batch_size, shuffle=True)

D_losses, G_losses = [], []


for epoch in tqdm(range(num_epochs)):
    running_loss_G, running_loss_D, running_PSNR = 0, 0, 0
    for i, (lr_images, hr_images) in enumerate(trainLoader):
        lr_images = lr_images.to(device)
        hr_images = hr_images.to(device)
        # Train the generator
        optimizer_G.zero_grad()

        sr_images = generator(lr_images)

        adversarial_loss = criterion_GAN(discriminator(sr_images), torch.ones(sr_images.size(0), 1).to(device))
        content_loss = criterion_content(sr_images, hr_images)

        total_loss_G = adversarial_loss + 0.01 * content_loss
        total_loss_G.backward()
        optimizer_G.step()

        # Train the discriminator
        optimizer_D.zero_grad()

        real_loss = criterion_GAN(discriminator(hr_images), torch.ones(hr_images.size(0), 1).to(device))
        fake_loss = criterion_GAN(discriminator(sr_images.detach()), torch.zeros(sr_images.size(0), 1).to(device))

        total_loss_D = (real_loss + fake_loss) / 2

        total_loss_D.backward()
        optimizer_D.step()

        # Print the loss every 10 iterations
        if i % 100 == 0:
            print('Epoch [{}/{}], Batch [{}/{}], Generator Loss: {:.4f}, Discriminator Loss: {:.4f}'
              .format(epoch+1, num_epochs, i+1, len(trainLoader), total_loss_G.item(), total_loss_D.item()))
        
        running_loss_D += total_loss_D.item()
        running_loss_G += total_loss_G.item()

    epoch_loss_D = running_loss_D / len(trainLoader)
    epoch_loss_G = running_loss_G / len(trainLoader)
    
    D_losses.append(epoch_loss_D)
    G_losses.append(epoch_loss_G)
    
    # print(f'Average Generator Loss: {G_losses:.4f}, Average Discriminator Loss: {D_losses:.4f}')
    # Save the model every 10 epochs
    if (epoch+1) % 10 == 0:
        torch.save(generator.state_dict(), 'generator_epoch{}.pth'.format(epoch+1))
        torch.save(discriminator.state_dict(), 'discriminator_epoch{}.pth'.format(epoch+1))
    if epoch == 0 or (epoch+1) % 10 == 0:
        with torch.no_grad():
            generator.eval()
            # for i, (lr_images, hr_images) in enumerate(valLoader):
#            lr_image, hr_image = next(iter(valLoader))
#            lr_image = lr_image.to(device)
#            hr_image = hr_image.to(device)
#
#            sr_image = generator(lr_image)
#            example_image = torch.cat([lr_image, hr_image, sr_image], dim=3)
#            example_image = make_grid(example_image, normalize=True, scale_each=True)
#            example_image = np.transpose(example_image.cpu().detach().numpy(), (1, 2, 0))
#            fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(15,5))
#            axs.imshow(example_image)
#            axs.set_title('Comparison of Images')
#            plt.show()

            # Print example images
            # print("Example Images:")
            # print("Low Resolution Image:")
            # imshow(lr_image)
            # print("High Resolution Image:")
            # imshow(hr_image)
            # print("Super Resolution Image:")
            # imshow(sr_image)
                
#                 for k in range(sr_images.size(0)):
#                     # Save the generated image
#                     save_image(sr_images[k], 'generated_images/epoch{}_batch{}_{}.png'.format(epoch+1, i+1, k+1))

#                     # Save the ground truth image
#                     save_image(hr_images[k], 'generated_images/epoch{}_batch{}_{}_gt.png'.format(epoch+1, i+1, k+1))

#                     # Save the low resolution image
#                     save_image(lr_images[k], 'generated_images/epoch{}_batch{}_{}_lr.png'.format(epoch+1, i+1, k+1))

#                 break

            generator.train()