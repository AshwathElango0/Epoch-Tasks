import cv2          #Necessary imports
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class ocr_model(nn.Module):         #Defining CNN class to make use of saved model state dictionary
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

#Loading and using saved state dictionary to build a model for inference
model_path = r"C:\Users\achus\Desktop\Epoch projetcs\ocr_model.pth"       #Change path to where model is saved
cnn_model = ocr_model()
state_dict = torch.load(model_path, map_location='cpu')
cnn_model.load_state_dict(state_dict)
cnn_model.eval()

class LSTMClassifier(nn.Module):        #Defining sentiment analysis model class to make use of saved state dictionary
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        x = self.fc(output[:, -1, :])
        return self.fc2(x)

#loading in saved vocabulary and state dictionary
vocab_path = r"C:\Users\achus\Desktop\Epoch projetcs\vocab.pth"     #Change to where vocabulary is saved
vocab = torch.load(vocab_path)
model_path = r"C:\Users\achus\Desktop\Epoch projetcs\senti_model.pth"   #Change to path of state dictionary
senti_model = LSTMClassifier(embed_size=100, num_classes=3, vocab_size=len(vocab), hidden_size=128, num_layers=4)
state_dict = torch.load(model_path, map_location='cpu')
senti_model.load_state_dict(state_dict)
senti_model.eval()

def encode_text(text, vocab):
    """Method is used to strip sentence of spaces, and create a list of characters encoded as per the vocabulary"""
    text = text.replace(" ", "")
    return [vocab.get(word, 0) for word in text]

def process_sentence(sentence, vocab, max_length):
    """Sentences are either padded with 0s or truncated to reach the max_length"""
    encoded_sentence = encode_text(sentence, vocab)
    if len(encoded_sentence) < max_length:
        encoded_sentence += [vocab['<PAD>']] * (max_length - len(encoded_sentence))
    else:
        encoded_sentence = encoded_sentence[:max_length]
    return torch.tensor(encoded_sentence).unsqueeze(0)  #Adding batch dimension to meet model input shape requirements

def classify_sentence(model, sentence, vocab, max_length):
    with torch.no_grad():  #No need to compute gradients as model is used for inference
        processed_sentence = process_sentence(sentence, vocab, max_length)
        logits = model(processed_sentence)
        output = F.softmax(logits, 1)           #Applying softmax function on predicted logits
        prediction = torch.argmax(output, dim=1).item()     #Predicted label is obtained
    sentiment = ['Angry', 'Happy', 'Neutral'][prediction]
    return sentiment

def preprocess_character_image(image):
    """Character image is reshaped and resized to match the model's input requirements"""
    image = image.astype(np.float32)
    resized_image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_CUBIC)
    normalized_image = resized_image / 255.0
    tensor_image = torch.tensor(normalized_image, dtype=torch.float32)  #Converting to tensor
    tensor_image = tensor_image.unsqueeze(0).unsqueeze(0)  #Adding batch and channel dimensions
    return tensor_image
    

def predict_character(image):
    """Character is predicted from the preprocessed image"""
    with torch.no_grad():
        logits = cnn_model(image)
        output = F.softmax(logits, 1)       #Prediction probabilities are obtained
        _, predicted_idx = torch.max(output, 1)     #Predicted label is obtained
        predicted_character = chr(predicted_idx.item() + ord('A'))  #Label is converted to character
    return predicted_character

def find_bounds(projection, threshold):
        """The bright regions in the projection(ROIs containing characters) are identified.
        The method used is histogram projection. Bright regions represent lines/characters/words, and dark regions are spaces"""
        bounds = []
        region = False
        for i, value in enumerate(projection):
            if value > threshold and not region:
                start = i
                region = True
            elif value <= threshold and region:
                end = i
                bounds.append((start, end))
                region = False
        if region:
            bounds.append((start, len(projection)))
        return bounds

def segment_chars(image):
    _, thresh = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY)    #Image is binarized using thresholding
    horizontal_projection = np.sum(thresh, axis=1)      #Summing up pixel values along rows
    horizontal_bounds = find_bounds(horizontal_projection, threshold=100)   #Finding bounds of bright regions(used to identify lines of text)

    cropped_lines = []      #Separating lines of characters and adding them to a list
    for (h_start, h_end) in horizontal_bounds:
        line = thresh[h_start:h_end, :]
        cropped_lines.append(line)
    cropped_images = []

    for line in cropped_lines:
        vertical_projection = np.sum(line, axis=0)      #Summing up pixel values along columns(within each currently active row)
        vertical_bounds = find_bounds(vertical_projection, threshold=100)   #Identifying characters regions

        for i, (v_start, v_end) in enumerate(vertical_bounds):
            if (i>0 and v_start-vertical_bounds[i-1][1]>40) or (i==0 and v_start>50):   #Accounting for spaces before the first word in the line and spaces between words in a line
                dummy_image = np.zeros_like(line[:, v_start:v_end])     #Add a pure black image(placeholder for spaces)
                cropped_images.append(dummy_image)
            cropped = line[:, v_start:v_end]

            cropped_images.append(cropped)      #Adding image of character

            if (i==len(vertical_bounds)-1 and 512-v_end>50):     #Adding placeholders if the word is the last word of the line and has a space after it
                dummy_image = np.zeros_like(line[:, v_start:v_end])
                cropped_images.append(dummy_image)
            
    return cropped_images       #Returning an array of images of characters and placeholders for spaces



def recognize_sentence(image):
    character_images = segment_chars(image)
    sentence = ''
    for image in character_images:
        if np.all(image == 0):  #Check if the image is a placeholder, and add a space if it is
            sentence += ' '
        else:
            image = preprocess_character_image(image)   #If the image is of a character, predict its value using the CNN and add it to the sentence
            output = predict_character(image)
            sentence += output
    return sentence

if __name__ == '__main__':
    image_path = r"C:\Users\achus\Downloads\line_6.png"     #Change to path of image to analyse
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  #Load image in grayscale format
    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_CUBIC)    #Reisizng with cubic interpolation to preserve details  

    if image is None:
        print(f"Failed to load image at path: {image_path}")
    else:
        recognized_sentence = recognize_sentence(image)     #Function to return the sentence identified
        print("Recognized Sentence:", recognized_sentence)

        sentiment = classify_sentence(senti_model, recognized_sentence, vocab, 128) #Funtion to identify the sentiment
        print(f"Sentiment: {sentiment}")
