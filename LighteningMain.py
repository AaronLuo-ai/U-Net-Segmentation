from utils.Lightening_object import LitModel
from utils.train_loader import TrainDataset
from utils.test_loader import TestDataset
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from pathlib import Path


def main():
    batch_path = Path("C:\\Users\\aaron.l\\Documents\\U-Net-Segmentation\\Data\\batch.csv")
    root_path = Path("C:\\Users\\aaron.l\\Documents\\U-Net-Segmentation\\Data")
    train_data = MRIDataset(batch_path, root_path)
    test_data = TestDataset()
    train_loader = DataLoader(train_data, batch_size=3, shuffle=True)  # Adjust batch size as needed
    val_loader = DataLoader(test_data, batch_size=3, shuffle=False)

    # Set up WandB logger
    wandb_logger = WandbLogger(project='Tumor Segmentation', log_model='all')

    checkpoint_callback = ModelCheckpoint(monitor="val_accuracy", mode="max")

    model = LitModel()
    print("type(model): ",type(model    ))
    # trainer = Trainer(logger=wandb_logger)
    trainer = Trainer()
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    main()