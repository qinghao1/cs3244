from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, Dropout
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

class Model:

	@staticmethod
	def compile(self, height, width, channels):

		inputs = Input(height, width, channels)

		# Layers here

		model = Model(input=inputs, outputs=last_layer)

		model.compile(optimizer = Nadam(), loss='categorical_crossentropy', metrics=['accuracy'])

		return model