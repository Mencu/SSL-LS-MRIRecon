import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import torch
import pytorch_lightning as pl

from dataloader import KspaceDataModule
from module import CascadeModule
from fft import NUFFT_v3, NUFFTAdj_v3


# Loggers and monitors
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor

PROJECT_NAME = 'supervised_test'
TB_PATH = '<your_path_here>'
# WANDB_PATH = '<your_path_here>'
LOG = True

tensorboard_logger = TensorBoardLogger(save_dir=TB_PATH, name=PROJECT_NAME)
# wandb_logger = WandbLogger(save_dir=WANDB_PATH, name=PROJECT_NAME, project="")
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
    'test': torch.nn.Identity()
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

test_dataloader = datamodule.test_dataloader()

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


# CHKPT_PATH = 'supervised_uncollapsed_D5C5_fix_fullRun_Self-supervised Physical Guided MRI Reconstruction/30_21lyho73/checkpoints/epoch=17-val_loss=0.00-other_metric=0.00.ckpt'

# DC fixed!
CHKPT_PATH = f'<your_path_here>'
model = CascadeModule(**hyperparameters) #.load_from_checkpoint(CHKPT_PATH)
print(f'Model loaded')


# Define triaing parameters
NUM_EPOCHS = 1000
ACCELERATOR = "gpu"
DEVICES = "auto"

trainer = pl.Trainer(max_epochs=NUM_EPOCHS,
                    accelerator=ACCELERATOR, devices=DEVICES,
                    logger=tensorboard_logger,
                    # strategy=DDPStrategy(find_unused_parameters=False)
                    )

results = trainer.test(model, dataloaders=test_dataloader, ckpt_path=CHKPT_PATH, verbose=True)

print(f'Results: {results}')

