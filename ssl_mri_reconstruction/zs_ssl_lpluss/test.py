import os 
import torch

torch.set_float32_matmul_precision('medium')

import pytorch_lightning as pl

from dataloader import KspaceDataModule
from module import SSLModule
from fft import NUFFT_v3, NUFFTAdj_v3


# Loggers and monitors
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor

# PROJECT_NAME = 'zs_ssl_lpluss_ssl'
PROJECT_NAME = 'zs_ssl_lpluss_ssl_1cycle'

TB_PATH = '<your_path_here>'
WANDB_PATH = '<your_path_here>'
LOG = True

tensorboard_logger = TensorBoardLogger(save_dir=TB_PATH, name=PROJECT_NAME)
# wandb_logger = WandbLogger(save_dir=WANDB_PATH, name=PROJECT_NAME, project="Self-supervised Physical Guided MRI Reconstruction", entity="sslftw")
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

test_dataloader = datamodule.test_dataloader()

# Define model
hyperparameters = {
    'n_ch': 2,          # number of channels
    'nc' : 10,           # number of iterations
    'nd' : 5,           # number of ResNet convs
    'lr': 5e-4,
    'op' : op,
    'op_adj' : op_adj,
    'config': nufft_config,
    'self_supervised': SELF_SUPERVISED,
    'gif_root': GIF_ROOT,
    'gif_root_test': GIF_ROOT_TEST, 
    'pred_path':PRED_PATH,
    'pd_path':PD_PATH,
}

CHKPT_PATH = f'<your_path_here>'

model = SSLModule(**hyperparameters)
print(f'Model loaded')

# Define triaing parameters
NUM_EPOCHS = 1000
ACCELERATOR = "gpu"
DEVICES = 1

# Define Trainer and start the training 
trainer = pl.Trainer(max_epochs=NUM_EPOCHS,
                    accelerator=ACCELERATOR, devices=DEVICES,
                    )

results = trainer.test(model, dataloaders=test_dataloader, ckpt_path=CHKPT_PATH, verbose=True)

print(f'Results: {results}')