    # CycleGAN
    # Copyright (C) 2026  Richard Qin

    # This program is free software: you can redistribute it and/or modify
    # it under the terms of the GNU General Public License as published by
    # the Free Software Foundation, either version 3 of the License, or
    # (at your option) any later version.

    # This program is distributed in the hope that it will be useful,
    # but WITHOUT ANY WARRANTY; without even the implied warranty of
    # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    # GNU General Public License for more details.

    # You should have received a copy of the GNU General Public License
    # along with this program.  If not, see <https://www.gnu.org/licenses/>.


# Dataset
import os
if os.path.exists("/kaggle/input/competitions/gan-getting-started"):
    is_kaggle = True
    ROOT = "/kaggle/input/competitions/gan-getting-started"
else:
    is_kaggle = False
    ROOT = os.getcwd()
RANK = int(os.environ.get("RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
DISTRIBUTED = WORLD_SIZE > 1

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class MonetPhotoDataset(Dataset):
    def __init__(self, monet_dir: str, photo_dir: str, transform=None):
        super().__init__()
        self.monet_dir = monet_dir
        self.photo_dir = photo_dir
        self.transform = transform

        self.monet_images = sorted(os.listdir(monet_dir))
        self.photo_images = sorted(os.listdir(photo_dir))

        self.monet_len = len(self.monet_images)
        self.photo_len = len(self.photo_images)

        self.length_dataset = max(self.monet_len, self.photo_len)

        print("Loading Monet images into memory...")
        self.monet_cache = []
        for img_name in tqdm(self.monet_images, desc="Loading Monet"):
            img_path = os.path.join(self.monet_dir, img_name)
            self.monet_cache.append(Image.open(img_path).convert("RGB"))

        print("Loading Photo images into memory...")
        self.photo_cache = []
        for img_name in tqdm(self.photo_images, desc="Loading Photo"):
            img_path = os.path.join(self.photo_dir, img_name)
            self.photo_cache.append(Image.open(img_path).convert("RGB"))
        print("Loaded images into memory successfully.")

    def __len__(self):
        return self.length_dataset
    
    def __getitem__(self, idx : int) -> tuple:
        # monet_img = self.monet_images[idx % self.monet_len]
        # photo_img = self.photo_images[idx % self.photo_len]

        # monet_path = os.path.join(self.monet_dir, monet_img)
        # photo_path = os.path.join(self.photo_dir, photo_img)

        # monet_image = Image.open(monet_path).convert("RGB")
        # photo_image = Image.open(photo_path).convert("RGB")

        # Use cached images to avoid redundant disk I/O
        monet_image = self.monet_cache[idx % self.monet_len]
        photo_image = self.photo_cache[idx % self.photo_len]

        if self.transform:
            monet_image = self.transform(monet_image)
            photo_image = self.transform(photo_image)

        return monet_image, photo_image

# Preprocessing
import torchvision.transforms as transforms
# Train transform
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
# Aditional transforms for data augmentation
# train_transform = transforms.Compose([
#     transforms.Resize(286),
#     transforms.RandomCrop(256),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomRotation(10),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ])
# Test transform
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Verify the dataset
from matplotlib import pyplot as plt
monet_dir = os.path.join(ROOT, "monet_jpg")
photo_dir = os.path.join(ROOT, "photo_jpg")


dataset = MonetPhotoDataset(monet_dir, photo_dir, transform=train_transform)


debug_loader = DataLoader(
    dataset, 
    batch_size=6, 
    shuffle=True, 
    num_workers=4, 
    pin_memory=True, 
    # persistent_workers=True, 
    # prefetch_factor=2, 
    drop_last=True
    )


if RANK == 0:
    for monet_images, photo_images in debug_loader:
        print(monet_images.shape)  # Should be [6, 3, 256, 256]
        print(photo_images.shape)  # Should be [6, 3, 256, 256]
        break

    X, y = dataset[0]
    print(X.shape, y.shape)  # Should be [3, 256, 256] [3, 256, 256]
    plt.subplot(1, 2, 1)
    plt.imshow((X.permute(1, 2, 0) * 0.5 + 0.5).numpy())  # Denormalize for visualization
    plt.title("Monet Image")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow((y.permute(1, 2, 0) * 0.5 + 0.5).numpy())  # Denormalize for visualization
    plt.title("Photo Image")
    plt.axis("off")
    # plt.show()

import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels : int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(in_channels)
        )
        return None
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return x + self.block(x)
    
class Generator(nn.Module):
    def __init__(self, input_shape: tuple, num_residual_blocks: int = 9, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        channels, height, width = input_shape

        # Initial Convolution and Downsampling
        out_features = 64
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels, out_features, kernel_size=7, stride=1, padding=0),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True)
        )

        in_features = out_features
        for _ in range(2):
            out_features *= 2
            self.initial.add_module(
                f"downsample_{out_features}",
                nn.Sequential(
                    nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                    nn.InstanceNorm2d(out_features),
                    nn.ReLU(inplace=True)
                )
            )
            in_features = out_features
        
        # Residual Blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(out_features) for _ in range(num_residual_blocks)]
        )

        # Upsampling
        for _ in range(2):
            out_features //= 2
            self.residual_blocks.add_module(
                f"upsample_{out_features}",
                nn.Sequential(
                    nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.InstanceNorm2d(out_features),
                    nn.ReLU(inplace=True)
                )
            )
            in_features = out_features
        
        # Final Convolution
        self.final = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(out_features, channels, kernel_size=7, stride=1, padding=0),
            nn.Tanh()
        )

        self.model = nn.Sequential(
            self.initial,
            self.residual_blocks,
            self.final
        )
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
# Test the generator
G = Generator(input_shape=(3, 256, 256))
test_result = G(torch.randn(6, 3, 256, 256))  # Add batch dimension
print("Input shape:", (6, 3, 256, 256))
print("Output shape:", test_result.shape)  # Should be [6, 3, 256, 256]

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_shape: tuple, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        channels, height, width = input_shape

        def discriminator_block(in_filters : int, out_filters : int, kernel_size : int = 4, stride : int = 2, padding : int = 1, normalize : bool = True) -> nn.Sequential:
            block = nn.Sequential(
                nn.Conv2d(in_filters, out_filters, kernel_size, stride, padding),
                nn.InstanceNorm2d(out_filters) if normalize else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True)
            )
            return block
            
        
        self.model = nn.Sequential(
            discriminator_block(channels, 64, stride=2, normalize=False),
            discriminator_block(64, 128, stride=2),
            discriminator_block(128, 256, stride=2),
            discriminator_block(256, 512, stride=2),

            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, kernel_size=4, padding=1)
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
# Test the discriminator
D = Discriminator(input_shape=(3, 256, 256))
test_result = D(torch.randn(6, 3, 256, 256))  # Add batch dimension
print("Input shape:", (6, 3, 256, 256))
print("Output shape:", test_result.shape)  # Should be [6, 1, 16, 16]

# Loss functions
import itertools
from torch.optim import Adam

criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

# loss weight hyperparameters
lambda_cycle = 10.0
lambda_identity = 5.0
# Formula: L_G = L_GAN + lambda_cycle * L_cycle + lambda_identity * L_identity

from torch.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from typing import Optional, Union

scaler_G = GradScaler()
scaler_D_M = GradScaler()
scaler_D_P = GradScaler()

# Distributed training setup
def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA.")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    else:
        rank = 0
        local_rank = 0

    return rank, local_rank, world_size, distributed

def cleanup_distributed(distributed: bool):
    if distributed and dist.is_initialized():
        dist.destroy_process_group()

def is_main_process(rank: int) -> bool:
    return rank == 0

# Train function for one epoch
def train_one_epoch(
    dataloader : DataLoader,
    generator_M2P : Union[Generator, DDP],
    generator_P2M : Union[Generator, DDP],
    discriminator_M : Union[Discriminator, DDP],
    discriminator_P : Union[Discriminator, DDP],
    optimizer_G : Adam,
    optimizer_D_M : Adam,
    optimizer_D_P : Adam,
    scaler_G : GradScaler,
    scaler_D_M : GradScaler,
    scaler_D_P : GradScaler,
    criterion_GAN : nn.Module,
    criterion_cycle : nn.Module,
    criterion_identity : nn.Module,
    device : torch.device,
    lambda_cycle : float,
    lambda_identity : float,
    show_progress : bool = True,
) -> tuple[float, float]:
    generator_M2P.train()
    generator_P2M.train()
    discriminator_M.train()
    discriminator_P.train()

    total_loss_G = 0.0
    total_loss_D = 0.0

    pbar = tqdm(dataloader, desc="Training Epoch", disable=not show_progress)

    for i, batch in enumerate(pbar):
        real_monet, real_photo = batch
        real_monet = real_monet.to(device)
        real_photo = real_photo.to(device)
        pred_real_M = discriminator_M(real_monet)
        valid_label = torch.ones_like(pred_real_M, device=device)
        fake_label = torch.zeros_like(pred_real_M, device=device)

        #-------------------------
        # Train Generators
        # -------------------------
        optimizer_G.zero_grad(set_to_none=True)

        with autocast('cuda'):
            # Identity loss
            identity_monet = generator_P2M(real_monet)
            loss_identity_monet = criterion_identity(identity_monet, real_monet) * lambda_identity

            identity_photo = generator_M2P(real_photo)
            loss_identity_photo = criterion_identity(identity_photo, real_photo) * lambda_identity
            # GAN loss
            fake_monet = generator_P2M(real_photo)
            loss_GAN_P2M = criterion_GAN(discriminator_M(fake_monet), valid_label)

            fake_photo = generator_M2P(real_monet)
            loss_GAN_M2P = criterion_GAN(discriminator_P(fake_photo), valid_label)
            # Cycle consistency loss
            recov_monet = generator_P2M(fake_photo)
            loss_cycle_monet = criterion_cycle(recov_monet, real_monet) * lambda_cycle

            recov_photo = generator_M2P(fake_monet)
            loss_cycle_photo = criterion_cycle(recov_photo, real_photo) * lambda_cycle

            # Total generator loss and backward
            loss_G = loss_identity_monet + loss_identity_photo + loss_GAN_P2M + loss_GAN_M2P + loss_cycle_monet + loss_cycle_photo
        
        scaler_G.scale(loss_G).backward()
        scaler_G.step(optimizer_G)
        scaler_G.update()

        #-------------------------
        # Train Discriminators
        # -------------------------
        optimizer_D_M.zero_grad(set_to_none=True)
        with autocast('cuda'):
            loss_real_M = criterion_GAN(discriminator_M(real_monet), valid_label)
            loss_fake_M = criterion_GAN(discriminator_M(fake_monet.detach()), fake_label)
            loss_D_M = (loss_real_M + loss_fake_M) * 0.5
        scaler_D_M.scale(loss_D_M).backward()
        scaler_D_M.step(optimizer_D_M)
        scaler_D_M.update()

        optimizer_D_P.zero_grad(set_to_none=True)
        with autocast('cuda'):
            loss_real_P = criterion_GAN(discriminator_P(real_photo), valid_label)
            loss_fake_P = criterion_GAN(discriminator_P(fake_photo.detach()), fake_label)
            loss_D_P = (loss_real_P + loss_fake_P) * 0.5
        scaler_D_P.scale(loss_D_P).backward()
        scaler_D_P.step(optimizer_D_P)
        scaler_D_P.update()

        total_loss_G += loss_G.item()
        total_loss_D += loss_D_M.item() + loss_D_P.item()

        pbar.set_postfix({
            "loss_G": f"{loss_G.item():.4f}",
            "loss_D": f"{(loss_D_M.item() + loss_D_P.item()):.4f}"
        })

    return total_loss_G / len(dataloader), total_loss_D / len(dataloader)

import torchvision.utils as vutils
def evaluate_visually(
    generator_M2P : Union[Generator, DDP],
    generator_P2M : Union[Generator, DDP],
    dataloader : DataLoader,
    device : torch.device,
    num_images : int = 4
) -> None:
    generator_M2P.eval()
    generator_P2M.eval()

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            real_monet, real_photo = batch
            real_monet = real_monet.to(device)
            real_photo = real_photo.to(device)

            fake_monet = generator_P2M(real_photo)
            fake_photo = generator_M2P(real_monet)

            # Create a grid of images for visualization
            img_grid = vutils.make_grid(
                torch.cat((real_monet[:num_images], fake_photo[:num_images], real_photo[:num_images], fake_monet[:num_images]), dim=0),
                nrow=num_images,
                normalize=True,
                scale_each=False,
                value_range=(-1, 1)
            )

            plt.figure(figsize=(12, 6))
            plt.imshow(img_grid.permute(1, 2, 0).cpu().numpy())
            plt.title(
                "Row 1: Real Monet  |  Row 2: Fake Photo (Generated)\n"
                "Row 3: Real Photo  |  Row 4: Fake Monet (Generated)",
                fontsize=14, pad=20
            )
            plt.axis("off")
            plt.show()
            break
    generator_M2P.train()
    generator_P2M.train()
    return None

import torch.backends.cudnn as cudnn
cudnn.benchmark = True  # Enable cuDNN auto-tuner for better performance

# Initialize models, optimizers, and loss functions
rank, local_rank, world_size, distributed = setup_distributed()
if distributed:
    device = torch.device(f"cuda:{local_rank}")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gpu_count = torch.cuda.device_count()
if is_main_process(rank):
    print(f"Current computing device in use: {device}")
    if device.type == 'cuda':
        print(f"Number of available GPUs detected: {gpu_count}")

train_sampler: Optional[DistributedSampler] = (
    DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    if distributed
    else None
)
dataloader = DataLoader(
    dataset,
    batch_size=10,
    shuffle=(train_sampler is None),
    sampler=train_sampler,
    num_workers=2,
    pin_memory=True,
    # persistent_workers=True,
    # prefetch_factor=2,
    drop_last=True,
)

generator_M2P = Generator(input_shape=(3, 256, 256)).to(device)
generator_P2M = Generator(input_shape=(3, 256, 256)).to(device)
discriminator_M = Discriminator(input_shape=(3, 256, 256)).to(device)
discriminator_P = Discriminator(input_shape=(3, 256, 256)).to(device)

if distributed:
    if is_main_process(rank):
        print("Using DistributedDataParallel for multi-GPU training.")
    generator_M2P = DDP(generator_M2P, device_ids=[local_rank], output_device=local_rank)
    generator_P2M = DDP(generator_P2M, device_ids=[local_rank], output_device=local_rank)
    discriminator_M = DDP(discriminator_M, device_ids=[local_rank], output_device=local_rank)
    discriminator_P = DDP(discriminator_P, device_ids=[local_rank], output_device=local_rank)

optimizer_G = Adam(itertools.chain(generator_M2P.parameters(), generator_P2M.parameters()), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_M = Adam(discriminator_M.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_P = Adam(discriminator_P.parameters(), lr=0.0002, betas=(0.5, 0.999))

def save_model(model, filepath):
    if isinstance(model, DDP):
        torch.save(model.module.state_dict(), filepath)
    else:
        torch.save(model.state_dict(), filepath)

# Train the model
num_epochs = 30
for epoch in range(num_epochs):
    if distributed and train_sampler is not None:
        train_sampler.set_epoch(epoch)

    avg_loss_G, avg_loss_D = train_one_epoch(
        dataloader,
        generator_M2P,
        generator_P2M,
        discriminator_M,
        discriminator_P,
        optimizer_G,
        optimizer_D_M,
        optimizer_D_P,
        scaler_G,
        scaler_D_M,
        scaler_D_P,
        criterion_GAN,
        criterion_cycle,
        criterion_identity,
        device,
        lambda_cycle,
        lambda_identity,
        show_progress=is_main_process(rank)
    )
    if is_main_process(rank):
        print(f"Epoch [{epoch+1}/{num_epochs}] - Average Loss G: {avg_loss_G:.4f}, Average Loss D: {avg_loss_D:.4f}")

    # Evaluate visually every 5 epochs
    if is_main_process(rank) and (epoch + 1) % 5 == 0:
        evaluate_visually(generator_M2P, generator_P2M, dataloader, device)

    # Save model checkpoints every 5 epochs
    # if (epoch + 1) % 5 == 0:
    #     save_model(generator_M2P, f"generator_M2P_epoch_{epoch+1}.pth")
    #     save_model(generator_P2M, f"generator_P2M_epoch_{epoch+1}.pth")
    #     save_model(discriminator_M, f"discriminator_M_epoch_{epoch+1}.pth")
    #     save_model(discriminator_P, f"discriminator_P_epoch_{epoch+1}.pth")

# Test Dataset and Dataloader
class TestMonetPhotoDataset(Dataset):
    def __init__(self, photo_dir: str, transform=None) -> None:
        super().__init__()
        self.photo_dir = photo_dir
        self.transform = transform

        self.photo_images = sorted(os.listdir(photo_dir))

    def __len__(self) -> int:
        return len(self.photo_images)
    
    def __getitem__(self, idx : int) -> torch.Tensor:
        photo_name = self.photo_images[idx]
        photo_path = os.path.join(self.photo_dir, photo_name)
        photo_image = Image.open(photo_path).convert("RGB")

        if self.transform:
            photo_image = self.transform(photo_image)
        else:
            photo_image = transforms.ToTensor()(photo_image)

        return photo_image  # Return image and filename for saving results

# Inference function
def translate_image(
    generator_P2M : Union[Generator, DDP],
    photo_dataloader : DataLoader,
    device : torch.device,
    output_dir : str = "./generated_images",
    max_images : int = 7038
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    generator_P2M.eval()
    img_count = 0

    with torch.no_grad():
        for batch in tqdm(photo_dataloader, desc="Translating to Monet Style"):
            real_photos = batch
            real_photos = real_photos.to(device)

            fake_monets = generator_P2M(real_photos)

            fake_monets = (fake_monets * 0.5) + 0.5 
            fake_monets = (fake_monets * 255).clamp(0, 255).to(torch.uint8)
            fake_monets = fake_monets.cpu().numpy()

            for i in range(fake_monets.shape[0]):
                img_array = fake_monets[i].transpose(1, 2, 0)
                img = Image.fromarray(img_array)
                img.save(os.path.join(output_dir, f"generated_{img_count:04d}.jpg"), format="JPEG")

                img_count += 1
                if img_count >= max_images:
                    break
            if img_count >= max_images:
                break

# create zip file of generated images
import zipfile
def create_zip_from_directory(directory : str, zip_name : str) -> None:
    with zipfile.ZipFile(zip_name, 'w') as zipf:
        for foldername, subfolders, filenames in os.walk(directory):
            for filename in filenames:
                file_path = os.path.join(foldername, filename)
                zipf.write(file_path, os.path.relpath(file_path, directory))

    return None

if is_kaggle:
    generate_img_dir = "/kaggle/working/generated_images"
else:
    generate_img_dir = "./generated_images"
if is_main_process(rank):
    test_dataset = TestMonetPhotoDataset(photo_dir=photo_dir, transform=test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    translate_image(generator_P2M, test_dataloader, device=device, output_dir=generate_img_dir, max_images=7038)
    create_zip_from_directory(generate_img_dir, "images.zip")

cleanup_distributed(distributed)