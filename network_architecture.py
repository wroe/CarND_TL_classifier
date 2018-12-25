from keras import backend as K

from keras.models import Sequential
from keras.layers import Convolution2D, Cropping2D, Dense, Dropout, Flatten, MaxPooling2D, Activation, Lambda
from keras.callbacks import ModelCheckpoint

#Define the model
def network_architecture():

    model = Sequential()

    model.add(Cropping2D(cropping=((0, 0), (0, 0)), \
        input_shape=(600, 800, 3)))
    model.add(Lambda(lambda x: x/127.5 - 1))

    model.add(Convolution2D (32, 8, strides=(4, 4), padding="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(2, 2))

    model.add(Convolution2D (64, 4, strides=(2, 2), padding="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(2, 2))

    model.add(Flatten())

    model.add(Dense(128, init='he_normal'))
    model.add(Activation('relu'))

    model.add(Dropout(0.3))

    model.add(Dense(32, init='he_normal'))
    model.add(Activation('relu'))

    model.add(Dense(4))

    model.add(Activation('softmax'))
    #model.add(Lambda(lambda x: K.tf.nn.softmax(x)))
    
    return(model)