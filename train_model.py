import os
import numpy as np
import pandas as pd
import cv2

import sklearn
from sklearn import model_selection

from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam

from network_architecture import *
from augment import *


nEpochs = 15
batch_size = 4

sim = False
if sim:
	MODEL_FILE_NAME = './sim_model.h5'
	trainfile = 'sim_train.csv'
	colormap = {'green':0, 'red':1, 'yellow':2, 'not_light':3}
else:
	MODEL_FILE_NAME = './site_model.h5'
	trainfile = 'site_train.csv'
	colormap = {'not_red':0, 'red':1, 'not_light':3}


# generator function to return images batchwise
def generator(samples, batch_size, apply_augment=True):
	n_samples, dummy = samples.shape

	while True:
		samples = sklearn.utils.shuffle(samples)
		for offset in range(0, n_samples-batch_size, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images = []
			colors = []

			for index, sample in batch_samples.iterrows():
				img = cv2.imread(sample.path)
				color = sample.color
				if apply_augment:
					img = augment(img)

				images.append(img)  # hard coded to center img
				colors.append(colormap[color])

			x_train = np.array(images)
			y_train = np.array(colors)
			y_train = to_categorical(y_train, num_classes=4)

			yield sklearn.utils.shuffle(x_train, y_train)

if __name__ == "__main__":

	data_set = pd.read_csv(os.path.join('./' + trainfile))

	# Split data into random training and validation sets
	x_train, x_valid = model_selection.train_test_split(data_set, test_size=.2)

	train_gen = generator(x_train, batch_size)
	validation_gen = generator(x_valid, batch_size/2, False)

	model = network_architecture()
	model.compile(optimizer=Adam(lr=5e-4), loss='categorical_crossentropy', metrics=['accuracy'])

	# checkpoint to save best weights after each epoch    based on the improvement in val_loss
	checkpoint = ModelCheckpoint(MODEL_FILE_NAME, monitor='val_loss', verbose=1,save_best_only=True, mode='min',save_weights_only=False)
	callbacks_list = [checkpoint]

	print('Simulator Model Training started....')

	history = model.fit_generator(
		train_gen, steps_per_epoch=len(x_train)//batch_size,
		epochs=nEpochs, validation_data=validation_gen,
		validation_steps=len(x_valid)//batch_size,
		verbose=1, callbacks=callbacks_list
	)

	# Destroying the current TF graph to avoid clutter from old models / layers
	K.clear_session()
