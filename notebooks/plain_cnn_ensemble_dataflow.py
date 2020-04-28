import tensorflow as tf
import numpy as np

class ensemble_cnn:

	def __init__(self):
		pass

	
	def predict(X, use='aug'):
		n_of_classes = 10
		n_of_samples = X.shape[0]

		y_sum = np.zeros((n_of_samples, n_of_classes))


		if (use=='aug' or use=='both'):
			for model_number in range(1, 4):
			    for iteration in range(1, 11):
			        model_name = f'model_{model_number}_{iteration}'
			        
			        model = tf.keras.models.load_model(f'./models/with_aug/{model_name}.h5')
			        
			        y_proba = model.predict(X)
			        y_hat = np.zeros_like(y_proba)
			        y_hat[np.arange(y_proba.shape[0]), np.argmax(y_proba, axis=1)] = 1

			        y_sum = y_sum + y_hat
			        
			        model=None

		if (use=='no_aug' or use=='both'):
			for model_number in range(1, 4):
			    for iteration in range(1, 11):
			        model_name = f'model_{model_number}_{iteration}'
			        
			        model = tf.keras.models.load_model(f'./models/no_aug/{model_name}.h5')
			        
			        y_proba = model.predict(X)
			        y_hat = np.zeros_like(y_proba)
			        y_hat[np.arange(y_proba.shape[0]), np.argmax(y_proba, axis=1)] = 1

			        y_sum = y_sum + y_hat
			        
			        model=None


		y_sum = y_sum + np.random.uniform(-0.1, 0.1, y_sum.shape)
		y_hat_final = np.zeros_like(y_sum)
		y_hat_final[np.arange(y_sum.shape[0]), np.argmax(y_sum, axis=1)] = 1

		return y_hat_final




