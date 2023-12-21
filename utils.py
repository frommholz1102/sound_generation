import torch
import torch.nn as nn
import torch.nn.functional as F

# Custom VAE Loss for VAE

class VAE_Loss(nn.Module):
    """
    Loss Function for Variational Autoencoder
    is combination of reconstruction loss and KL divergence loss

    loss = weight * reconstruction_loss + kl_loss

    recon_loss_weight (float):  weight for reconstruction loss, determines how much
                                we want to prioritize reconstruction loss over KL divergence loss
    """

    def __init__(self, recon_loss_weight):
        super(VAE_Loss, self).__init__()
        self.recon_loss_weight = recon_loss_weight

    def forward(self, y_target, y_predicted, mu, logvar):
        # calculate both losses separately
        reconstruction_loss = self._calculate_reconstruction_loss(y_target, y_predicted)
        kl_loss = self._calculate_kl_loss(mu, logvar)
        # combine losses (include weight for reconstruction loss)
        combined_loss = self.recon_loss_weight * reconstruction_loss + kl_loss
        return combined_loss

    def _calculate_reconstruction_loss(self, y_target, y_predicted):
        # Simple MSE loss
        error = y_target - y_predicted
        reconstruction_loss = torch.mean(torch.square(error), axis=[1, 2, 3])
        return reconstruction_loss

    def _calculate_kl_loss(self, mu, logvar):
        # Kullback-Leibler divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kl_loss