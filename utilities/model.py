from keras.applications import DenseNet121
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.models import Model
import numpy as np

class CNNModel:
    def __init__(self):
        self.__model = self.get_model()
    
    def get_model(self):
        base = DenseNet121(
        include_top=False,
        input_shape = (136, 136, 3),
        weights=None
        )
        x = base.output
        x = Flatten()(x)
        x = Dense(512, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(7, activation="softmax")(x)
        return Model(inputs=[base.input], outputs = [x])
    
    def load(self, file_path):
        self.__model.load_weights(file_path)
        
    def predict(self, heatmap):
        return np.argmax(self.__model.predict(np.expand_dims(heatmap, 0))[0])