import utils
import wandb
import torch
import numpy as np
import datetime as dt

from tqdm import tqdm
from BaseModel import BaseModel
from VICRegLoss import VICRegLoss
from models import Encoder, Projector


class VICRegModel(BaseModel):
    MODEL_NAME = 'VICReg.pth'
    PROJECT_NAME = 'AML-Ex2-VICReg'
    RUN_NAME = f'{MODEL_NAME.split(".")[0]}_{dt.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}'
    TRAINED_MODELS_DIR = 'trained_models'

    # training parameters
    LEARNING_RATE = 3e-4
    TRAIN_EPOCHS_NUM = 30
    BETAS = (0.9, 0.999)
    WEIGHT_DECAY = 1e-6
    LOG_INTERVAL = 10

    # model parameters
    ENC_DIM = 128
    PROG_DIM = 512

    # loss parameters
    GAMMA = 1
    EPSILON = 1e-4

    INV_LOSS_WEIGHT = 25
    VAR_LOSS_WEIGHT = 25
    CONV_LOSS_WEIGHT = 1

    def __init__(self, learning_rate=LEARNING_RATE,
                 eps=EPSILON, gamma=GAMMA, w_inv=INV_LOSS_WEIGHT,
                 w_var=VAR_LOSS_WEIGHT, w_conv=CONV_LOSS_WEIGHT):
        super(VICRegModel, self).__init__()
        self.config = {'device': self.device,
                       'eps': eps,
                       'gamma': gamma,
                       'w_inv': w_inv,
                       'w_var': w_var,
                       'w_conv': w_conv,
                       'learning_rate': learning_rate,
                       'train_epochs_num': self.TRAIN_EPOCHS_NUM}

        self.encoder = Encoder(D=self.ENC_DIM)
        self.decoder = Projector(D=self.ENC_DIM, proj_dim=self.PROG_DIM)

        self.criterion = VICRegLoss(eps=eps, gamma=gamma, w_inv=w_inv, w_var=w_var, w_conv=w_conv)
        self.to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, betas=self.BETAS,
                                          weight_decay=self.WEIGHT_DECAY)

    def forward(self, batch, deploy=False):
        if self.training and not deploy:
            z1 = self.decoder(self.encoder(batch[0].to(self.device)))
            z2 = self.decoder(self.encoder(batch[1].to(self.device)))
            return z1, z2

        img_batch = (batch[0].unsqueeze(0) if len(batch[0].shape) == 3 else batch[0]).to(self.device)
        return self.encoder(img_batch)

    def fit(self, train_loader, test_loader, epoch_num=TRAIN_EPOCHS_NUM, log_interval=LOG_INTERVAL, save_model=True):
        self.train()
        print(f'Training {self.MODEL_NAME.split(".")[0]}...')
        wandb.login(key=utils.get_wandb_key())
        wandb.init(project=self.PROJECT_NAME, name=self.RUN_NAME)
        n = len(train_loader.dataset)
        for epoch in np.arange(epoch_num):
            running_loss = 0.0
            print(f'Epoch {epoch + 1}/{epoch_num}:')
            for i, batch in enumerate(train_loader):
                self.optimizer.zero_grad()
                z1, z2 = self.forward(batch)
                loss = self.criterion(z1, z2)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

                if not i % log_interval:
                    print(f'Train Epoch: {epoch} [{i * len(batch[0])}/{n} '
                          f'({100. * i / n:.0f}%)]\tLoss: {loss.item():.6f}')
                    running_loss = 0.0

            with torch.no_grad():
                test_losses = [self.criterion(*self.forward(batch), test=True) for batch in tqdm(test_loader)]
                self.criterion.log_test_loss(test_losses)

        print('Finished Training')
        if save_model:
            self.save_model()
        wandb.finish()
