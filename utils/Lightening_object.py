from models.network_unet import UNet
import torch.optim as optim
from utils.loss import DiceLoss
import pytorch_lightning as pl

class LitModel(pl.LightningModule):
    def __init__(self, lr=0.01):
        super().__init__()
        self.learning_rate = lr
        self.save_hyperparameters()
        self.model = UNet()
        self.loss = DiceLoss()

    def setup(self, stage=None):
        # we set up only relevant datasets when stage is specified
        if stage == 'fit' or stage is None:
            mnist = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist, [55000, 5000])
        if stage == 'test' or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def forward(self, x):
        return self.model(x)

    def loss(self, outputs, targets):
        return DiceLoss(outputs, targets)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))
        return optimizer

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        inputs = inputs.permute(1, 0, 2, 3)
        labels = labels.permute(1, 0, 2, 3)
        outputs = self.model(inputs)
        loss = self.loss(outputs, labels)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.loss(outputs, labels)
        self.log('val_loss', loss, on_step=True, on_epoch=True)
        return loss
