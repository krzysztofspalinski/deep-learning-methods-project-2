import tensorflow as tf
from tensorflow.keras import regularizers


class model_1:

	

	def __init__(self):
		pass


	def get_model():

		NUM_CLASSES = 10
		INPUT_SHAPE = (32, 32, 3)

		model = tf.keras.Sequential()
		model.add(tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=INPUT_SHAPE, kernel_regularizer=regularizers.l2(0.0005)))
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.MaxPooling2D((2, 2)))

		model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=INPUT_SHAPE, kernel_regularizer=regularizers.l2(0.001)))
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.MaxPooling2D((2, 2)))

		model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.MaxPooling2D((2, 2)))

		model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.MaxPooling2D((2, 2)))

		model.add(tf.keras.layers.Flatten())

		model.add(tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
		model.add(tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))

		model.add(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'))

		return model


	def describe():
		tmp_model = model_1.get_model()
		tmp_model.summary()


class model_2:

	

	def __init__(self):
		pass


	def get_model():

		NUM_CLASSES = 10
		INPUT_SHAPE = (32, 32, 3)

		model = tf.keras.Sequential()
		model.add(tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=INPUT_SHAPE, kernel_regularizer=regularizers.l2(0.0005)))
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.MaxPooling2D((2, 2)))

		model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=INPUT_SHAPE, kernel_regularizer=regularizers.l2(0.001)))
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.MaxPooling2D((2, 2)))

		model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.MaxPooling2D((2, 2)))

		model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.MaxPooling2D((2, 2)))

		model.add(tf.keras.layers.Conv2D(256, (2, 2), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.Conv2D(512, (2, 2), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.MaxPooling2D((2, 2)))

		model.add(tf.keras.layers.Flatten())

		model.add(tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
		model.add(tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
		model.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))

		model.add(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'))

		return model


	def describe():
		tmp_model = model_2.get_model()
		tmp_model.summary()



class model_3:

	

	def __init__(self):
		pass


	def get_model():

		NUM_CLASSES = 10
		INPUT_SHAPE = (32, 32, 3)

		model = tf.keras.Sequential()
		model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=INPUT_SHAPE, kernel_regularizer=regularizers.l2(0.001)))
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.MaxPooling2D((2, 2)))

		model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.MaxPooling2D((2, 2)))

		model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.MaxPooling2D((2, 2)))

		model.add(tf.keras.layers.Conv2D(256, (2, 2), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.Conv2D(256, (2, 2), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.MaxPooling2D((2, 2)))

		model.add(tf.keras.layers.Conv2D(512, (2, 2), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.Conv2D(512, (2, 2), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.MaxPooling2D((2, 2)))

		model.add(tf.keras.layers.Flatten())

		model.add(tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
		model.add(tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))

		model.add(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'))

		return model


	def describe():
		tmp_model = model_3.get_model()
		tmp_model.summary()