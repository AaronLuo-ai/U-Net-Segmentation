from models.network_unet import UNet
from utils import loss
from utils.train_loader import TrainDataset
from utils.test_loader import TestDataset
from torch.utils.data import DataLoader
from utils.loss import CustomCrossEntropyLoss
import torch.nn as nn
import wandb
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
from PIL import Image
import io
import tempfile

def main():
    train_data = TrainDataset()
    test_data = TestDataset()
    train_loader = DataLoader(train_data, batch_size = 4, shuffle = True)
    test_loader = DataLoader(test_data, batch_size = 3, shuffle = True)
    num_epoch = 1
    model = UNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to('cuda')  # or model.cuda()
    loss_func = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    train_loss = []
    test_loss = []
    wandb.login(key="f0e77a676faded347d56b7af7737bd9d349c1e7a")
    wandb.init(project="U-Net Segmentation", name="U-Net Segmentation")
    wandb.watch(model, log="all")

    for epoch in range(num_epoch):
        print(f'Epoch {epoch + 1}/{num_epoch}')

        epoch_train_loss = 0
        model.train()

        for i, train_batch in enumerate(train_loader):
            # calculate outputs
            inputs, labels = train_batch
            inputs = inputs.to('cuda')  # or input_tensor.cuda()
            labels = labels.to('cuda')  # or target_tensor.cuda()
            inputs = inputs.float()
            labels = labels.float()
            # print("inputs.shape: ", inputs.shape)
            # print("labels.shape: ", labels.shape)
            # print("inputs.dtype: ", inputs.dtype)
            # print("labels.dtype: ", labels.dtype)


            # for i in range(inputs.shape[0]):
            #     plt.figure(figsize=(10, 6))  # Create a figure with a grid of subplots
            #     plt.imshow(inputs[i].squeeze(), cmap='gray')
            #     plt.tight_layout()  # Adjust layout to avoid overlap
            #     plt.title(f'Image {i + 1}')
            #     plt.show()

            # for i in range(labels.shape[0]):
            #     plt.figure(figsize=(10, 6))  # Create a figure with a grid of subplots
            #     plt.imshow(labels[i].squeeze(), cmap='gray')
            #     plt.tight_layout()  # Adjust layout to avoid overlap
            #     plt.title(f'Mask {i + 1}')
            #     plt.show()

            outputs = model(inputs) # accepts (N, channel, width, length)
            outputs = (outputs - outputs.min()) / (outputs.max() - outputs.min())
            #for i in range(outputs.shape[0]):
                #plt.figure(figsize=(10, 6))  # Create a figure with a grid of subplots
                #plt.imshow(outputs[i].detach().numpy().squeeze(), cmap='gray')
                #plt.tight_layout()  # Adjust layout to avoid overlap
                #plt.title(f'outputs {i + 1}')
                #plt.show()
            # print("max of outputs, min of outputs: ", outputs.max(), outputs.min())
            # print("max of labels, min of labels: ",labels.max(), labels.min())
            loss = loss_func(outputs, labels)
            loss.requires_grad_(True)
            # print("Loss: ", loss)

            # optimizing network
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            # record metrics
            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_loss.append(avg_train_loss)
        print("avg_train_loss: ",avg_train_loss)
        wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss})

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for j, test_batch in enumerate(test_loader):
                inputs, labels = train_batch
                inputs = inputs.to('cuda')  # or input_tensor.cuda()
                labels = labels.to('cuda')  # or target_tensor.cuda()
                inputs = inputs.float()
                labels = labels.float()

                loss = loss_func(outputs, labels)
                epoch_val_loss += loss.item()


        avg_val_loss = epoch_val_loss / len(test_loader)
        test_loss.append(avg_val_loss)
        print("avg_val_loss: ",avg_val_loss)
        wandb.log({"epoch": epoch + 1, "val_loss": avg_val_loss})

    # Now plot the graph after the loop
    plt.plot(range(1, num_epoch + 1), train_loss, label='Training Loss')
    plt.plot(range(1, num_epoch + 1), test_loss, label='Test Loss')
    plt.xlabel('Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.show()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    wandb.log({'chart': wandb.Image(Image.open(buf))})
    plt.close()

if __name__ == '__main__':
    main()
