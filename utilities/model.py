from tensorflow.keras.models import load_model
import numpy as np

class Model:
    def __init__(self):
        self.__model = None
    
    def load(self, file_path):
        self.__model = load_model(file_path)
        
    def predict(self, heatmap):
        return np.argmax(self.__model.predict([heatmap])[0])