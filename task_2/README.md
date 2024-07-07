This folder has a collection of files described below:

1. OCR.py -> This is the main program which performs OCR and sentiment analysis.
2. cv_model.py -> This contains the source code for the building and training of the CNN used for OCR.
3. nlp_model.py -> This contains the source code for the building and training of the LSTM used for sentiment analysis.
4. ocr_model.pth -> This is the state dictionary of the CNN trained using cv_model.py. It is used in OCR.py.
5. senti_model.pth -> This is the state dictionary of the LSTM trained using nlp_model.py. It is used in OCR.py.
6. vocab.pth -> This is the vocabulary of the sentiment analysis model. It is used in OCR.py.

NOTE: OCR.py contains paths to files 4, 5 and 6 as mentioned above. In order to run the code successfully, the paths need to be changed to where the files are stored on your device.

It also uses paths to the image that is to be analysed. This also requires to be changed to the path of the desired image.
