from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten


def DQN(input_channels, input_size, output_size):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(4, 4), strides=4, activation='relu',
                     input_shape=(input_size, input_size, input_channels)))
    model.add(Conv2D(filters=64, kernel_size=(4, 4), strides=2, activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=1, activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(output_size, activation=None))
    return model
