!pip install pytorch_lightning

import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader,random_split
from tqdm import tqdm

import pytorch_lightning as pl
import torchmetrics
from torchmetrics import Metric

import config
from model import NN
from dataset import MNISTDataModule


model = NN(input_size=config.INPUT_SIZE, learning_rate=config.LEARNING_RATE, num_classes=config.NUM_CLASSES)


dm = MNISTDataModule(
  data_dir=config.DATA_DIR,
  batch_size=config.BATCH_SIZE,
  num_workers=config.NUM_WORKERS
)


trainer = pl.Trainer(accelerator=config.ACCELERATOR, devices=config.DEVICES, max_epochs=3, precision=config.PRECISION) 
trainer.fit(model, dm)
trainer.validate(model, dm)
trainer.test(model, dm)