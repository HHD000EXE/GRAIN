import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys


class TrainDataset(Dataset):
    def __init__(self, image_folder, image_folder2, image_folder3, label_folder, transform=None):
        self.image_folder = image_folder
        self.image_folder2 = image_folder2
        self.image_folder3 = image_folder3
        self.label_folder = label_folder
        self.transform = transform
        self.image_files = os.listdir(image_folder)
        self.image_files2 = os.listdir(image_folder2)
        self.image_files3 = os.listdir(image_folder3)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.image_files[idx])
        img_name2 = os.path.join(self.image_folder2, self.image_files2[idx])
        img_name3 = os.path.join(self.image_folder3, self.image_files3[idx])
        image = Image.open(img_name)
        image2 = Image.open(img_name2)
        image3 = Image.open(img_name3)

        label_file_path = os.listdir(label_folder)[0]
        label_data = pd.read_csv(os.path.join(label_folder, label_file_path))  # Load the label data from the associated CSV file
        # Assuming label is in the second column
        if label_data.iloc[idx, 0][:-10] == label_data.iloc[idx+1, 0][:-10]:
            label = np.array(label_data.iloc[idx+1, 1:3], dtype="float32")
        else:
            label = np.array(label_data.iloc[idx, 1:3], dtype="float32")

        if self.transform:
            image = self.transform(image)
            image2 = self.transform(image2)
            image3 = self.transform(image3)
        # Convert NumPy array to a PyTorch tensor
        concatenated_image = np.concatenate((image, image2, image3), axis=0)
        concatenated_image = torch.from_numpy(concatenated_image).to(device)
        label = torch.from_numpy(label).to(device)
        return concatenated_image.to(device), label.to(device)


class TestDataset(Dataset):
    def __init__(self, image_folder, image_folder2, image_folder3, label_folder, transform=None):
        self.image_folder = image_folder
        self.image_folder2 = image_folder2
        self.image_folder3 = image_folder3
        self.label_folder = label_folder
        self.transform = transform
        self.image_files = os.listdir(image_folder)
        self.image_files2 = os.listdir(image_folder2)
        self.image_files3 = os.listdir(image_folder3)
        self.label_file_path = os.listdir(label_folder)[0]
        self.label_data = pd.read_csv(os.path.join(label_folder, self.label_file_path))  # Load the label data from the associated CSV file

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.image_files[idx])
        img_name2 = os.path.join(self.image_folder2, self.image_files2[idx])
        img_name3 = os.path.join(self.image_folder3, self.image_files3[idx])
        image = Image.open(img_name)
        image2 = Image.open(img_name2)
        image3 = Image.open(img_name3)

        # Assuming label is in the second column
        if self.label_data.iloc[idx, 0][:-10] == self.label_data.iloc[idx+1, 0][:-10]:
            label = np.array(self.label_data.iloc[idx+1, 1:3], dtype="float32")
        else:
            label = np.array(self.label_data.iloc[idx, 1:3], dtype="float32")

        if self.transform:
            image = self.transform(image)
            image2 = self.transform(image2)
            image3 = self.transform(image3)
        # Convert NumPy array to a PyTorch tensor
        concatenated_image = np.concatenate((image, image2, image3), axis=0)
        concatenated_image = torch.from_numpy(concatenated_image).to(device)
        label = torch.from_numpy(label).to(device)
        return concatenated_image.to(device), label.to(device)


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, emb_size):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, emb_size, H/P, W/P]
        x = x.flatten(2)  # [B, emb_size, n_patches]
        x = x.transpose(1, 2)  # [B, n_patches, emb_size]
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_size, n_heads, ff_hidden_mult, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(emb_size, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)

        self.ff = nn.Sequential(
            nn.Linear(emb_size, ff_hidden_mult * emb_size),
            nn.GELU(),
            nn.Linear(ff_hidden_mult * emb_size, emb_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm1(x)
        attn, _ = self.attention(query=x, key=x, value=x)
        x = self.dropout(x)
        x = self.norm2(attn + x)
        ff = self.ff(x)
        x = ff + x
        return x


class MLPHead_classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPHead_classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # First fully connected layer
        self.activation = nn.GELU()                  # Activation function
        self.fc2 = nn.Linear(hidden_dim, output_dim) # Final layer that outputs class scores

    def forward(self, x):
        x = self.fc1(x)      # Input goes through the first fully connected layer
        x = self.activation(x)  # Apply activation function
        x = self.fc2(x)      # Final layer to produce the output
        return x


class MLPHead_regression(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPHead_regression, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # First fully connected layer
        self.activation = nn.GELU()                  # Activation function
        self.fc2 = nn.Linear(hidden_dim, output_dim) # Final layer that outputs class scores

    def forward(self, x):
        x = self.fc1(x)      # Input goes through the first fully connected layer
        x = self.activation(x)  # Apply activation function
        x = self.fc2(x)      # Final layer to produce the output
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=128, patch_size=16, in_channels=7, outputcells=2, emb_size=768, depth=6, n_heads=6, ff_hidden_mult=4, dropout=0.1):
        super().__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, emb_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positional_embedding = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, emb_size))
        self.dropout = nn.Dropout(dropout)

        self.transformer_blocks = nn.Sequential(
            *[TransformerEncoderBlock(emb_size, n_heads, ff_hidden_mult, dropout) for _ in range(depth)]
        )

        self.to_cls_token = nn.Identity()
        self.mlp_head = MLPHead_regression(emb_size, 4*emb_size, outputcells)

    def forward(self, x):
        x = self.patch_embedding(x)
        b, t, _ = x.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.positional_embedding
        x = self.dropout(x)

        x = self.transformer_blocks(x)

        cls_token_final = self.to_cls_token(x[:, 0, :])
        return self.mlp_head(cls_token_final)

def train_epoch(model, train_loader, criterion, optimizer):
    model.train()
    train_error, train_total, train_total_loss = 0, 0, 0
    for data, targets in train_loader:
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_total += 1
        try:
            new_error = outputs - targets
            train_error = train_error + torch.abs(new_error)
        except:
            train_error = train_error
        train_total_loss += loss

    # Update total
    train_epoch_accuracy = train_error / train_total
    train_ave_loss = train_total_loss / train_total

    return train_epoch_accuracy, train_ave_loss

def evaluate(model, test_loader, criterion):
    model.eval()
    test_error, test_total_loss, test_total = 0, 0, 0
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)

            test_total += 1
            try:
                new_error = outputs - targets
                test_error = test_error + torch.abs(new_error)
            except:
                test_error = test_error
            test_total_loss += loss

        # Update total
        test_epoch_accuracy = test_error / test_total
        test_ave_loss = test_total_loss / test_total

    return test_epoch_accuracy, test_ave_loss


if __name__ =="__main__":
    # Check if CUDA is available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Instantiate the model and move it to the selected device
    model = VisionTransformer().to(device)
    print(model)

    # Transformations applied on each image
    transform = transforms.Compose([transforms.Resize((128, 128)),transforms.ToTensor(),])

    # Load your dataset
    image_folder = "train_images(D)"
    image_folder2 = "train_images(RawD)"
    image_folder3 = "train_images(Action)"
    label_folder = "label"
    image_folder_test = "Test_data/train_images(D)"
    image_folder_test2 = "Test_data/train_images(RawD)"
    image_folder_test3 = "Test_data/train_images(Action)"
    label_folder_test = "Test_data/label"
    train_dataset = TrainDataset(image_folder, image_folder2, image_folder3, label_folder, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
    test_dataset = TestDataset(image_folder_test, image_folder_test2, image_folder_test3, label_folder_test, transform=transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    # Loss function
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.L1Loss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.000005)

    # Number of training epochs
    epochs = 200
    train_loss_record, test_loss_record = [], []
    original_stdout = sys.stdout

    with open('output.txt', 'w') as f:
    # Training loop with accuracy
        for epoch in range(epochs):
            train_epoch_accuracy, train_ave_loss = train_epoch(model, train_loader, criterion, optimizer)
            test_epoch_accuracy, test_ave_loss = evaluate(model, test_loader, criterion)
            train_loss_record.append(train_ave_loss)
            test_loss_record.append(test_ave_loss)

            sys.stdout = original_stdout
            print("{:.2%}".format(epoch/float(epochs)))

            sys.stdout = f
            print("##########Epoch", epoch)
            print("Epoch average error(train)", train_epoch_accuracy)
            print("Average loss", train_ave_loss)
            print("Epoch average error(test)", test_epoch_accuracy)
            print("Average loss", test_ave_loss)

    # plot loss vs epoch
    train_loss_record_np = [loss.detach().cpu().numpy() for loss in train_loss_record]
    test_loss_record_np = [loss.detach().cpu().numpy() for loss in test_loss_record]
    plt.plot(train_loss_record_np, 'green', label='Train')
    plt.plot(test_loss_record_np, 'red', label='Test')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('loss vs epoch.png')

    # Save the trained model if needed
    torch.save(model.state_dict(), 'trained_model.pth')


