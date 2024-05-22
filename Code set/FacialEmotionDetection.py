from datetime import date
import os
import random

import numpy as np
import seaborn as sns
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix
# from keras.preprocessing.image import load_img
from PIL import Image


class EmotionDetector:
    def __init__(self, model_path: str, img_path: str):
        or_img_path = 'path/to/image.png'
        self.or_image = Image.open(img_path)
        self.IMG_HEIGHT = 48
        self.IMG_WIDTH = 48
        self.class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.validation_datagen = ImageDataGenerator(rescale=1. / 255)
        self.my_model = load_model(model_path, compile=False)
        self.img = Image.open(img_path).convert('L')  # 'L' mode converts to grayscale
        self.img = self.img.resize((self.IMG_HEIGHT, self.IMG_WIDTH))
        self.img_array = np.array(self.img) / 255.0
        self.img_array = np.expand_dims(self.img_array, axis=0)

    def predict(self):
        prediction = self.my_model.predict(self.img_array)
        predicted_label = self.class_labels[np.argmax(prediction)]
        # plt.imshow(self.or_image)
        # plt.title(f'Predicted Label: {predicted_label}')
        # plt.axis('off')
        # plt.show()
        return predicted_label

    def generate_information(self):
        start_date = date(2023, 1, 1).toordinal()
        end_date = date(2023, 12, 31).toordinal()
        random_day = date.fromordinal(random.randint(start_date, end_date))
        user_id = random.randint(1, 10)
        school_id = random.randint(1, 2)
        return random_day, user_id, school_id


# ed = EmotionDetector('emotion_detection_model_100epochs.h5', 'C:\\Users\\rcc\\Desktop\\image\\wf.jpg')
# ed.predict()
# ed.save_predictions()
