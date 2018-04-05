import keras
from keras.optimizers import Nadam

class Model:

	@staticmethod
	def getCompiledModel(model_type, image_dim, num_classes):

		model = None
		input_shape = image_dim + (3,)

		if model_type == "densenet":
			model = keras.applications.densenet.DenseNet121(include_top=True, weights=None, input_shape=input_shape, classes=num_classes)

		if model_type == "mobilenet":
			model = keras.applications.mobilenet.MobileNet(include_top=True, weights=None, input_shape=input_shape, classes=num_classes)

		if model_type == "inception-resnet":
			model = keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=True, weights=None, input_shape=input_shape, pooling=None, classes=num_classes)

		model.compile(optimizer = Nadam(), loss='categorical_crossentropy', metrics=['accuracy'])

		return model