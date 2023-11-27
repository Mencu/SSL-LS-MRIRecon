import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = '5'

import torch
import pytorch_lightning as pl

from dataloader import KspaceDataModule
from module import CascadeModule
from fft import NUFFT_v3, NUFFTAdj_v3


# Loggers and monitors
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor

from torchvision.transforms import RandomRotation

PROJECT_NAME = 'supervised_collapsed_D5C5_fullRun_DCPM'
# PROJECT_NAME = 'supervised_uncollapsed_D5C5_DCfix_augmentation_fullRun_DCPM'
TB_PATH = '<your_path_here>'
# WANDB_PATH = '<your_path_here>'
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
    'train': torch.nn.Identity(),
    'val': torch.nn.Identity(),
    'test': torch.nn.Identity(),
}

# Only this needed for training dataloader actually
ROOT = f'<your_path_here>'

# Leave rest of params here, just for information on the dataset used 
IMG_PATH = f'<your_path_here>'
LIMIT_NUM_SAMPLES = None
NUM_WORKERS = 8
SHUFFLE = False
COILS = [2, 3, 4, 5, 6, 7] # as in Wenqi's config
NUM_CARDIAC_CYCLES = 4
TRANSIENT_MAGNETIZATION = 600


datamodule = KspaceDataModule(BS, TRANSFORM, ROOT, IMG_PATH, LIMIT_NUM_SAMPLES, NUM_WORKERS, 
                    SHUFFLE, COILS, NUM_CARDIAC_CYCLES, TRANSIENT_MAGNETIZATION, 
                    op, op_adj, [nufft_config['img_dim'], nufft_config['img_dim']])

train_loader = datamodule.train_dataloader()
val_loader = datamodule.val_dataloader()

# Define model
hyperparameters = {
    'n_ch': 2,          # number of channels
    'nf' : 64,          # number of filters
    'ks' : 3,           # kernel size
    'nc' : 5,           # number of iterations
    'nd' : 5,           # number of CRNN/BCRNN/CNN layers
    'lr': 1e-5,
    'op' : op,
    'op_adj' : op_adj,
}

model = CascadeModule(**hyperparameters)

# Define triaing parameters
NUM_EPOCHS = 1000
ACCELERATOR = "gpu"
DEVICES = "auto"

CHKPT_PATH = f'<your_path_here>'

# Define monitors and callbacks
from pytorch_lightning.callbacks import ModelCheckpoint
checkpoint_callback = ModelCheckpoint(
            filename='{epoch}-{val_loss:.2f}-{other_metric:.2f}',
                monitor= 'epoch/val_loss',
                    save_top_k = 5)

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
early_stop_callback = EarlyStopping(monitor="epoch/val_loss", min_delta=0.00, patience=2, verbose=False, mode="min")


# Some general info about the training before starting it
print("Len training dataset : ", len(datamodule.train_dataset),
    "Batch Size : ", BS, "NUM_EPOCHS : ",NUM_EPOCHS )
print("Total training steps : ", len(datamodule.train_dataset)//BS*NUM_EPOCHS)

# Define Trainer and start the training 
trainer = pl.Trainer(max_epochs=NUM_EPOCHS,
                    accelerator=ACCELERATOR, devices=DEVICES,
                    callbacks=[checkpoint_callback, lr_monitor, early_stop_callback],
                    logger=[tensorboard_logger],#,wandb_logger],
                    log_every_n_steps=2,
                    # strategy=DDPStrategy(find_unused_parameters=False)
                    )

if __name__ == "__main__":
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader,
                ckpt_path=CHKPT_PATH,)

