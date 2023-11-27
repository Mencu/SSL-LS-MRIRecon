import os 
import typing_extensions
from importlib import reload
reload(typing_extensions)
import torch
import random
import pytorch_lightning as pl
import numpy as np

from dataloader import KspaceDataModule
from module import SSLModule
from fft import NUFFT_v3, NUFFTAdj_v3

# Loggers and monitors
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor

torch.set_float32_matmul_precision('medium')

seed = 43
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

PROJECT_NAME = 'zs_ssl_lpluss_ssl_1cycle'

TB_PATH = '<your_path_here>'
LOG = True

tensorboard_logger = TensorBoardLogger(save_dir=TB_PATH, name=PROJECT_NAME)
lr_monitor = LearningRateMonitor(logging_interval='step')


# Define NUFFT operators and configs
nufft_config = {
    'img_dim': 256,     # image dimension
    'osf': 1,         # oversampling factor
    'kernel_width': 3,
    'sector_width': 8,
}

op = NUFFT_v3(nufft_config)
op_adj = NUFFTAdj_v3(configs=nufft_config)

# Define datamodule specific parameters (config)
BS = 1
TRANSFORM = {
    'train': None,
    'val': None,
    'test': None,
}

# Choose dataset acceleration
ROOT =  f'<your_path_here>'


# Leave rest of params here, just for information on the dataset used 
IMG_PATH = f'<your_path_here>'

GIF_ROOT = '<your_path_here>'
GIF_ROOT_TEST = '<your_path_here>'

PRED_PATH = '<your_path_here>'
PD_PATH = '<your_path_here>'

LIMIT_NUM_SAMPLES = None
NUM_WORKERS = 8
SHUFFLE = False
SELF_SUPERVISED = True

datamodule = KspaceDataModule(BS, TRANSFORM, ROOT, IMG_PATH, LIMIT_NUM_SAMPLES, NUM_WORKERS, 
                    SHUFFLE, op, op_adj, [nufft_config['img_dim'], nufft_config['img_dim']], SELF_SUPERVISED)

train_loader = datamodule.train_dataloader()
val_loader = datamodule.val_dataloader()

# Define model
hyperparameters = {
    'n_ch': 2,          # number of channels
    'nc' : 10,           # number of iterations
    'nd' : 5,           # number of convs
    'lr': 1e-4,
    'op' : op,
    'op_adj' : op_adj,
    'config': nufft_config,
    'self_supervised': SELF_SUPERVISED,
    'gif_root': GIF_ROOT,
    'gif_root_test': GIF_ROOT_TEST, 
    'pred_path':PRED_PATH,
    'pd_path':PD_PATH,
}

model = SSLModule(**hyperparameters)

# Define triaing parameters
NUM_EPOCHS = 1000
ACCELERATOR = "gpu"
DEVICES = 1

# Define monitors and callbacks
from pytorch_lightning.callbacks import ModelCheckpoint
checkpoint_callback = ModelCheckpoint(
            filename='{epoch}-{val_loss:.2f}',
            monitor= 'epoch/val_loss',
            save_top_k = 3)

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
early_stop_callback = EarlyStopping(monitor="epoch/val_loss", min_delta=0.00, patience=50, verbose=False, mode="min")


# Some general info about the training before starting it
print(f'Starting training for {PROJECT_NAME}')
print("Len training dataset : ", len(datamodule.train_dataset),
    "Batch Size : ", BS, "NUM_EPOCHS : ",NUM_EPOCHS )
print("Total training steps : ", len(datamodule.train_dataset)//BS*NUM_EPOCHS)

# Define Trainer and start the training 
trainer = pl.Trainer(max_epochs=NUM_EPOCHS,
                    accelerator=ACCELERATOR, devices=DEVICES,
                    callbacks=[checkpoint_callback, lr_monitor, early_stop_callback],
                    logger=[tensorboard_logger],#wandb_logger],
                    log_every_n_steps=2,
                    num_sanity_val_steps=0,
                    detect_anomaly=True,
                    # track_grad_norm='inf',
                    # gradient_clip_val=0.8,
                    # gradient_clip_algorithm='norm'
                    # resume_from_checkpoint=CHKPT_PATH,
                    )

if __name__ == "__main__":
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

