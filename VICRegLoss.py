import torch
import wandb
import numpy as np
from torch import nn


class VICRegLoss(nn.Module):
    def __init__(self, eps, gamma, w_inv, w_var, w_conv):
        super(VICRegLoss, self).__init__()
        self.eps, self.gamma, self.w_inv, self.w_var, self.w_conv = eps, gamma, w_inv, w_var, w_conv

    @staticmethod
    def calc_inv_loss(z_batch, z_tag_batch):
        mse = torch.mean(torch.square(z_batch - z_tag_batch))
        return mse

    @staticmethod
    def calc_var_loss(z_batch, z_tag_batch, eps, gamma):
        l_z = VICRegLoss._calc_var_loss_single(z_batch, eps, gamma)
        l_z_tag = VICRegLoss._calc_var_loss_single(z_tag_batch, eps, gamma)
        return l_z + l_z_tag

    @staticmethod
    def _calc_var_loss_single(z, eps, gamma):
        sigma = torch.sqrt(torch.var(z, dim=0) + eps)
        l_z = torch.mean(torch.max(torch.zeros_like(sigma), gamma - sigma))
        return l_z

    @staticmethod
    def calc_cov_loss(z_batch, z_tag_batch):
        l_cov_z = VICRegLoss._calc_cov_loss_single(z_batch)
        l_cov_z_tag = VICRegLoss._calc_cov_loss_single(z_tag_batch)

        return l_cov_z + l_cov_z_tag

    @staticmethod
    def _calc_cov_loss_single(z):
        z_cov = torch.cov(z.T)
        l_cov_z = torch.square(z_cov - torch.diag(torch.diag(z_cov))).sum() / z_cov.shape[0]
        return l_cov_z

    def forward(self, predictions, target, test=False):
        inv_loss = self.calc_inv_loss(predictions, target)
        var_loss = self.calc_var_loss(predictions, target, self.eps, self.gamma)
        cov_loss = self.calc_cov_loss(predictions, target)
        loss = self.w_inv * inv_loss + self.w_var * var_loss + self.w_conv * cov_loss

        log_res = {'inv_loss': inv_loss, 'var_loss': var_loss, 'cov_loss': cov_loss, 'loss': loss}

        if test:
            return log_res

        wandb.log(log_res)
        return loss

    @staticmethod
    def log_test_loss(test_losses):
        log_res = {'test_loss': np.mean([x['loss'].cpu() for x in test_losses]),
                   'test_inv_loss': np.mean([x['inv_loss'].cpu() for x in test_losses]),
                   'test_var_loss': np.mean([x['var_loss'].cpu() for x in test_losses]),
                   'test_cov_loss': np.mean([x['cov_loss'].cpu() for x in test_losses])}
        wandb.log(log_res)
        print(f'Test loss: {log_res["test_loss"]:.3f}')
