import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

class MLP(nn.Module):
    def __init__(self, input_dim, layer_widths=[512, 128], output_dim=1, sigmoid=True):
        super(MLP, self).__init__()
        self.sigmoid = sigmoid
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, layer_widths[0]))
        self.layers.append(nn.ReLU())
        for i in range(len(layer_widths) - 1):
            self.layers.append(nn.Linear(layer_widths[i], layer_widths[i + 1]))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(layer_widths[-1], output_dim))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        if self.sigmoid:
            return torch.sigmoid(x)
        else:
            return torch.log_softmax(x, dim=1)

class LitModel(pl.LightningModule):
    def __init__(self, model, focal_loss=False, lr=0.0001):
        super(LitModel, self).__init__()
        self.model = model
        self.focal_loss = focal_loss
        self._lr = lr
        if focal_loss:
            raise ValueError("Focal loss not supported yet")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'train/loss')

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'val/loss')

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'test/loss')

    def eval_step(self, batch, batch_idx, log_name):
        x, y = batch
        y_hat = self(x)

        if self.model.sigmoid:
            loss = F.binary_cross_entropy(y_hat, y.float(), reduction='mean')
        else:
            loss = F.nll_loss(y_hat, torch.argmax(y, dim=1))
        self.log(log_name, loss, prog_bar=True if not self.focal_loss else False)  # Enable progress bar logging
        
        if not self.model.sigmoid: # calculate entropy
            probs = torch.exp(y_hat)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1).mean()
            self.log(f'{log_name}/entropy', entropy, prog_bar=True if not self.focal_loss else False)  # Log entropy
        
        return loss

    def change_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self._lr = lr

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self._lr)
        return self.optimizer


class LitSAE(pl.LightningModule):
    """
    F is the "feature space" dimension
    D is the "dictionary" dimension

    TODO: figure out how to train correctly with a leaky relu
    """
    def __init__(self, model_dim, dict_dim, sparsity_coefficient=1.0):
        super().__init__()
        self.encoder_DF = nn.Linear(model_dim, dict_dim)
        self.decoder_FD = nn.Linear(dict_dim, model_dim)
        nn.init.xavier_normal_(self.encoder_DF.weight)
        self.encoder_DF.bias.data = torch.ones_like(self.encoder_DF.bias.data)  # To reduce dead neurons
        nn.init.orthogonal_(self.decoder_FD.weight)
        self.decoder_FD.bias.data = torch.zeros_like(self.decoder_FD.bias.data)

        self.lda = sparsity_coefficient
        self.activated_codes_D = torch.zeros((dict_dim,))
        self.threshold = 1e-4
        self.epsilon = 1e-6

    def to(self, *args, **kwargs):
        self.activated_codes_D = self.activated_codes_D.to(*args, **kwargs)
        return super().to(*args, **kwargs)
    
    def encode(self, x_F):
        # pre_relu = self.encoder_DF(x_F)
        # print(f"Pre-ReLU mean: {pre_relu.mean():.3f}, std: {pre_relu.std():.3f}")
        # post_relu = nn.ReLU()(pre_relu)
        # print(f"Post-ReLU mean: {post_relu.mean():.3f}, std: {post_relu.std():.3f}")
        x_sparse_D = nn.ReLU()(self.encoder_DF(x_F - self.decoder_FD.bias))
        # x_sparse_D = nn.LeakyReLU(0.01)(self.encoder_DF(x_F))
        return x_sparse_D
    
    def decode(self, x_D):
        # Normalize the dictionary columns during the forward pass
        decoder_weights_FD = self.decoder_FD.weight
        norms_D = torch.norm(decoder_weights_FD, p=2, dim=0, keepdim=True)
        normalized_weights_FD = decoder_weights_FD / (norms_D + self.epsilon)
        return F.linear(x_D, normalized_weights_FD, self.decoder_FD.bias)

    def forward(self, x, stage=""):
        # x_sparse_D = self.encode(x)
        # x_hat_F = self.decode(x_sparse_D)
        # return x_sparse_D, x_hat_F
        x_sparse_D = self.encode(x)
        if stage == "":
            x_hat_F = self.decode(x_sparse_D)
            return x_sparse_D, x_hat_F
        else:
            x_sparse_D_mask = torch.zeros_like(x_sparse_D)
            if stage == "train":
                x_sparse_D_act = x_sparse_D[self.train_active_codes]
            elif stage == "val":
                x_sparse_D_act = x_sparse_D[self.val_active_codes]
            elif stage == "test":
                train_active = set(self.train_active_codes.tolist())
                val_active = set(self.val_active_codes.tolist())
                self.active_codes = torch.tensor(list(train_active | val_active)).to(x_sparse_D.device)
                x_sparse_D_act = x_sparse_D[:, self.active_codes]
            else:
                raise ValueError("Invalid stage")
            x_sparse_D_mask[:, self.train_active_codes] = x_sparse_D_act
            x_hat_F = self.decode(x_sparse_D_mask)
            return x_sparse_D_act, x_hat_F
    
    def step(self, batch, batch_idx, prefix):
        if len(batch) == 2:
            x_BF, y = batch
        elif len(batch) == 1:
            x_BF = batch
        else:
            raise ValueError("Invalid batch format")

        x_sparse_BD, x_hat_BF = self(x_BF)
        mse = F.mse_loss(x_hat_BF, x_BF)
        sparsity = torch.mean((x_sparse_BD > self.threshold).float().sum(dim=1))
        self.activated_codes_D = torch.sum((x_sparse_BD > self.threshold).float(), dim=0)
        loss = mse + self.lda * torch.norm(x_sparse_BD, p=1, dim=1).mean()

        self.log(f'{prefix}/loss', loss, prog_bar=True)
        self.log(f'{prefix}/reconstruction', mse, prog_bar=True)
        self.log(f'{prefix}/sparsity', sparsity, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f'{prefix}/dead_codes', len(self.activated_codes_D) - torch.sum(self.activated_codes_D > 0),
                 prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'train')

    def on_train_epoch_start(self):
        dead_encoder_codes_mask = torch.nonzero(self.activated_codes_D < self.threshold).squeeze()
        nn.init.xavier_normal_(self.encoder_DF.weight[dead_encoder_codes_mask])
        self.encoder_DF.bias.data[dead_encoder_codes_mask] = torch.ones_like(self.encoder_DF.bias.data[dead_encoder_codes_mask])
        return super().on_train_epoch_start()
    
    def on_train_epoch_end(self):
        self.train_dead_codes = torch.nonzero(self.activated_codes_D < self.threshold).squeeze()
        self.train_active_codes = torch.nonzero(self.activated_codes_D >= self.threshold).squeeze()
        return super().on_train_epoch_end()

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'val')

    def on_validation_epoch_end(self):
        self.val_dead_codes = torch.nonzero(self.activated_codes_D < self.threshold).squeeze()
        self.val_active_codes = torch.nonzero(self.activated_codes_D >= self.threshold).squeeze()
        return super().on_validation_epoch_end()

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'test')

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return self.optimizer

    def top_k(self, x_BF, k):
        # Get the top k dictionary elements
        dictionary_FD = self.decoder_FD.weight.data
        x_sparse_BD = self.encode(x_BF)
        top_k_indices_BK = torch.topk(x_sparse_BD, k, dim=1).indices
        top_k_indices_BKF = top_k_indices_BK.unsqueeze(2).expand(-1, -1, dictionary_FD.shape[1])
        top_k_values_BKF = torch.gather(dictionary_FD, 0, top_k_indices_BKF)
        return top_k_values_BKF
