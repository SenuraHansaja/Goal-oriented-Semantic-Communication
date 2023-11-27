from functools import partial
from typing import Sequence, Tuple, Union

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as VisionF
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy
from torchvision.datasets import CIFAR10
from torchvision.models.resnet import resnet34
from torchvision.utils import make_grid
from torchvision import models
from torchsummary import summary
from pytorch_lightning.loggers import CSVLogger
from tqdm import tqdm



logger = CSVLogger("logs_out", name="encoder_logs")


batch_size = 64
num_workers = 8
max_epochs = 200
z_dim = 1024


class BarlowTwinsTransform:
    def __init__(self, train=True, input_height=224, gaussian_blur=True, jitter_strength=1.0, normalize=None):
        self.input_height = input_height
        self.gaussian_blur = gaussian_blur
        self.jitter_strength = jitter_strength
        self.normalize = normalize
        self.train = train

        color_jitter = transforms.ColorJitter(
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.2 * self.jitter_strength,
        )

        color_transform = [transforms.RandomApply([color_jitter], p=0.8), transforms.RandomGrayscale(p=0.2)]

        if self.gaussian_blur:
            kernel_size = int(0.1 * self.input_height)
            if kernel_size % 2 == 0:
                kernel_size += 1

            color_transform.append(transforms.RandomApply([transforms.GaussianBlur(kernel_size=kernel_size)], p=0.5))

        self.color_transform = transforms.Compose(color_transform)

        if normalize is None:
            self.final_transform = transforms.ToTensor()
        else:
            self.final_transform = transforms.Compose([transforms.ToTensor(), normalize])

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(self.input_height),
                transforms.RandomHorizontalFlip(p=0.5),
                self.color_transform,
                self.final_transform,
            ]
        )

        self.finetune_transform = None
        if self.train:
            self.finetune_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.finetune_transform = transforms.ToTensor()

    def __call__(self, sample):
        return self.transform(sample), self.transform(sample), self.finetune_transform(sample)
    
def cifar10_normalization():
    normalize = transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
    )
    return normalize


train_transform = BarlowTwinsTransform(
    train=True, input_height=32, gaussian_blur=False, jitter_strength=0.5, normalize=cifar10_normalization()
)
train_dataset = CIFAR10(root=".", train=True, download=True, transform=train_transform)

val_transform = BarlowTwinsTransform(
    train=False, input_height=32, gaussian_blur=False, jitter_strength=0.5, normalize=cifar10_normalization()
)
val_dataset = CIFAR10(root=".", train=False, download=True, transform=train_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True,pin_memory = True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True, pin_memory = True)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=1024, output_dim=1024):
        super().__init__()

        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def forward(self, x):
        return self.projection_head(x)


def fn(warmup_steps, step):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    else:
        return 1.0


def linear_warmup_decay(warmup_steps):
    return partial(fn, warmup_steps)



'''loss function'''

class BarlowTwinsLoss(nn.Module):
    def __init__(self, batch_size, lambda_coeff=5e-3, z_dim=128):
        super().__init__()

        self.z_dim = z_dim
        self.batch_size = batch_size
        self.lambda_coeff = lambda_coeff

    def off_diagonal_ele(self, x):
        # taken from: https://github.com/facebookresearch/barlowtwins/blob/main/main.py
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z1, z2):
        # N x D, where N is the batch size and D is output dim of projection head
        z1_norm = (z1 - torch.mean(z1, dim=0)) / torch.std(z1, dim=0)
        z2_norm = (z2 - torch.mean(z2, dim=0)) / torch.std(z2, dim=0)

        cross_corr = torch.mm(z1_norm.T, z2_norm) / self.batch_size

        on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal_ele(cross_corr).pow_(2).sum()

        return on_diag + self.lambda_coeff * off_diag, on_diag, off_diag
    

''''encoder implementation'''

class BarlowTwins(LightningModule):
    def __init__(
        self,
        encoder,
        encoder_out_dim,
        num_training_samples,
        batch_size,
        lambda_coeff=5e-3,
        z_dim=128,
        learning_rate=1e-4,
        warmup_epochs=10,
        max_epochs=200,
    ):
        super().__init__()

        self.encoder = encoder
        self.projection_head = ProjectionHead(input_dim=encoder_out_dim, hidden_dim=encoder_out_dim, output_dim=z_dim)
        self.loss_fn = BarlowTwinsLoss(batch_size=batch_size, lambda_coeff=lambda_coeff, z_dim=z_dim)

        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

        self.train_iters_per_epoch = num_training_samples // batch_size

    def forward(self, x):
        return self.encoder(x)

    def shared_step(self, batch):
        (x1, x2, _), _ = batch

        z1 = self.projection_head(self.encoder(x1))
        z2 = self.projection_head(self.encoder(x2))

        loss, on_diag, off_diag= self.loss_fn(z1, z2)
       
        self.log("on_diag", on_diag.sum(), on_step=True, on_epoch=True, prog_bar= True, logger=logger)
        self.log("off_diag", off_diag.sum(), on_step=True, on_epoch=True, prog_bar= True, logger=logger)

        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        warmup_steps = self.train_iters_per_epoch * self.warmup_epochs

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(warmup_steps),
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]
    


    

'''this is the classifier implementation'''
class classifire(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=10),
        )
    
    def forward(self, x:torch.Tensor):
        x = self.block(x)
        return x

    
''' this is the encoder training part'''


encoder = resnet34()

# for CIFAR10, replace the first 7x7 conv with smaller 3x3 conv and remove the first maxpool
encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
encoder.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)

# replace classification fc layer of Resnet to obtain representations from the backbone
encoder.fc = nn.Identity()


encoder_out_dim = 512

model = BarlowTwins(
    encoder=encoder,
    encoder_out_dim=encoder_out_dim,
    num_training_samples=len(train_dataset),
    batch_size=batch_size,
    z_dim=z_dim,
)


checkpoint_callback = ModelCheckpoint(every_n_epochs=100, save_top_k=-1, save_last=True)

trainer = Trainer(
    max_epochs=max_epochs,
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
    callbacks=[#online_finetuner,
        checkpoint_callback],
    logger=logger
)

# uncomment this to train the model
# this is done for the tutorial so that the notebook compiles
trainer.fit(model, train_loader,)#val_loader)



'''this is the GoAI part'''

classifire_model = classifire()


ai_optimizer = torch.optim.Adam(classifire_model.parameters(), lr = 1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(ai_optimizer, 200)
goai_loss_fn = nn.CrossEntropyLoss()
# ai_optimizer = torch.optim.SGD(classifire_model.parameters(), lr=1e-3,
#                                 momentum=0.9, weight_decay=5e-4)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(ai_optimizer, 200)


''''training loop for the classifier'''
epochs =  100

for epoch in tqdm(range(epochs)):
    running_loss = 0.0
    running_corrects = 0
    for batch in train_loader:
        (img1, img2, _), label = batch
        img1 = img1.to('cuda')
        model = model.to('cuda')
        label = label.to('cuda')
        model.eval()
        classifire_model.to('cuda')
        classifire_model.train()

        encoder_out = model.forward(img1)
        input_to_goai = torch.unsqueeze(encoder_out, dim=1)             ### these unsuqeese is done to make the data follow NCHW pattern which is used inside the pytorch as a data format
        input_to_goai = torch.unsqueeze(input_to_goai, dim=1)
        #y_pred = GOAI_model(encoder_out)
        
       # GOAI_model.to('cuda')
        y_pred = classifire_model(input_to_goai)
        _, preds = torch.max(y_pred, 1)
        
        loss = goai_loss_fn(y_pred, label)
        ai_optimizer.zero_grad()

        loss.backward()

        ai_optimizer.step()
        running_loss += loss.item() * batch_size
        running_corrects += torch.sum(preds == label.data)
    scheduler.step()
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)

    print('Epoch: {} Loss: {:.4f} Acc: {:.4f}'.format(epoch,  epoch_loss, epoch_acc))


    ''''test accuracy '''


correct = 0
total = 0
with torch.inference_mode():
    for batch in val_loader:
         (img1, img2, _), label = batch
         model.eval()
         classifire_model.eval()
         model.to('cpu')
         classifire_model.to('cpu')
    
         encoder_out = model.forward(img1)
         input_to_goai = torch.unsqueeze(encoder_out, dim=1)             ### these unsuqeese is done to make the data follow NCHW pattern which is used inside the pytorch as a data format
         input_to_goai = torch.unsqueeze(input_to_goai, dim=1)
         y_pred = classifire_model(input_to_goai)
         _, preds = torch.max(y_pred.data, 1)
         total += label.size(0)
         correct += (preds == label).sum().item()
# epoch_acc = running_corrects.double() / len(val_loader.dataset)
# print('Acc: {:.4f}'.format(epoch_acc))
print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))     