import pandas as pd         #Necessary imports
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import numpy as np
from tqdm import tqdm

data = pd.read_csv(r"C:\Users\achus\Desktop\Epoch projetcs\sentiment_analysis_dataset.csv") #Reading data

class TextDataset(Dataset):     #Creating custom dataset class to use character-wise encoding
    def __init__(self, texts, labels, vocab=None):
        self.texts = texts
        self.labels = labels
        if vocab:
            self.vocab = vocab
        else:
            self.build_vocab()
    
    def build_vocab(self):
        all_chars = set()
        for text in self.texts:
            all_chars.update(text.replace(" ", ""))
        self.vocab = {char: idx+1 for idx, char in enumerate(sorted(all_chars))}
        self.vocab['<PAD>'] = 0        
    
    def encode_text(self, text):
        text = text.replace(" ", "")
        return [self.vocab.get(char, 0) for char in text]
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoded_text = self.encode_text(self.texts[idx])
        label = self.labels[idx]
        return torch.tensor(encoded_text), torch.tensor(label)

le = LabelEncoder()             #Encoding labels
labels = le.fit_transform(data['sentiment'])
train_texts, train_labels = data['line'], labels

train_dataset = TextDataset(train_texts.tolist(), train_labels.tolist())    #Preparing dataset(using all samples for training due to lack of data)

def collate_fn(batch):      #Using collate function to pad sentences
    texts, labels = zip(*batch)
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
    return padded_texts, torch.tensor(labels)

train_loader = DataLoader(train_dataset, batch_size=30, shuffle=True, collate_fn=collate_fn)    #Preparing data loader

class LSTMClassifier(nn.Module):    #Defining model class
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, bidirectional=True)  #Multi-layer bidirectional LSTM
        self.fc = nn.Linear(hidden_size*2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        x = self.fc(output[:, -1, :])
        return self.fc2(x)

#Defining vocab_size and some hyperparameters...since many layers share them, defining them outside is easier
vocab_size = len(train_dataset.vocab)
embed_size = 100
hidden_size = 128
num_classes = len(le.classes_)
num_layers = 4
num_epochs = 100

device = 'cuda' if torch.cuda.is_available() else 'cpu'     #Choosing training device
model = LSTMClassifier(vocab_size, embed_size, hidden_size, num_classes, num_layers).to(device) #Instantiating model
criterion = nn.CrossEntropyLoss()       #Choosing loss and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

for epoch in range(num_epochs):     #Training loop
    correct = 0
    total = 0
    with tqdm(train_loader, total=len(train_loader), unit='batch') as pb:       #Adding progress bar using tqdm
        model.train()
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(texts)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            pb.update(1)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = correct * 100 / total            #Calculating accuracy
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Accuracy: {accuracy}')

vocab = train_dataset.vocab

#Saving model state dictionary and vocab for future use
torch.save(vocab, r"C:\Users\achus\Desktop\Epoch projetcs\vocab.pth")
torch.save(model.state_dict(), r"C:\Users\achus\Desktop\Epoch projetcs\senti_model.pth")
