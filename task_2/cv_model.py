import torch            #Necessary imports
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from PIL import Image

df = pd.read_csv(r"C:\Users\achus\Downloads\alphabets_28x28.csv")       #Reading data

pixel_cols = [col for col in df.columns if col != 'label']          #Removing corrupted records
df[pixel_cols] = df[pixel_cols].apply(pd.to_numeric, errors='coerce')
df = df.dropna()

df = df.sample(frac=1, random_state=42).reset_index(drop=True)      #Shuffling data

labels = df['label'].to_numpy()         #Encoding labels
le = LabelEncoder()
int_labels = le.fit_transform(labels)

images = df[pixel_cols].values.reshape(-1, 1, 28, 28).astype(np.float32)        #Reshaping images to add batch and channel dimensions
images = np.multiply(images, 1/255.0)

train_size = int(0.8 * len(int_labels))     #Preparing to split data into traning and validation sets

class dataset_maker(Dataset):           #Custom dataset class which can apply transforms to the images
    def __init__(self, values, targets, transform=None):
        self.values = values
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.values)
    
    def __getitem__(self, idx):
        value = self.values[idx]
        target = self.targets[idx]

        #Squeezing to remove batch dimension
        image = value.squeeze()
        image = Image.fromarray((image * 255).astype(np.uint8), mode='L')

        if self.transform:      #Applying transform
            image = self.transform(image)

        #Converting to tensor and normalizing
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.tensor(image).unsqueeze(0)  #Adding channel dimension

        return image, target


transform = transforms.Compose([                #Defining transform to be used...to augment data
    transforms.RandomRotation(10),
])

#Preparing training and validation sets
train_data = dataset_maker(images[:train_size], int_labels[:train_size], transform)
train_data_loader = DataLoader(train_data, batch_size=128, shuffle=True)
val_data = dataset_maker(images[train_size:], int_labels[train_size:], transform)
val_data_loader = DataLoader(val_data, batch_size=128, shuffle=False)

class ocr_model(nn.Module):     #Defining custom CNN class...using VGG architecture(conv -> conv -> pooling)
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same')
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same')
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding='same')

        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(in_features=7*7*128, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=26)

    def forward(self, x):
        x = F.relu(self.conv2(F.relu(self.conv1(x))))
        x = self.pool(x)
        x = F.relu(self.conv4(F.relu(self.conv3(x))))
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   #Choosing device for training
model = ocr_model().to(device=device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)      #Preparing loss and optimizer
criterion = nn.CrossEntropyLoss()

num_epochs = 10         #Setting up custom training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    with tqdm(train_data_loader, total=len(train_data_loader), unit='batch') as pb:     #Adding a progress bar
        for data, target in train_data_loader:      #Iterating over training data
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            output = model(data)
            
            loss = criterion(output, target)
            
            loss.backward()
            
            optimizer.step()
            
            train_loss += loss.item()
            pb.update(1)
    train_loss /= len(train_data_loader)
    
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in val_data_loader:        #Iterating over validation data
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            
            loss = criterion(output, target)
            
            val_loss += loss.item()
            
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    val_loss /= len(val_data_loader)
    accuracy = 100.0 * correct / total      #Calculating accuracy
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {accuracy:.2f}%')

print('Training finished.')

#Saving model state dictionary for future use
torch.save(model.state_dict(), r"C:\Users\achus\Desktop\Epoch projetcs\ocr_model.pth")