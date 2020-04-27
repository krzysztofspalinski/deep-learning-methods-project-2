import tensorflow as tf

class ResnetIdentityBlock(tf.keras.Model):
  def __init__(self, kernel_size, filters, batch_normalization=True, conv_first=False):
    super(ResnetIdentityBlock, self).__init__(name='')
    
    self.residual_layers = []
    
    for i in range(len(filters)):
        
        if conv_first:
            setattr(self, 'conv' + str(i+1), tf.keras.layers.Conv2D(filters[i], kernel_size, padding='same'))
            self.residual_layers.append('conv' + str(i+1))

            if batch_normalization:
                setattr(self, 'bn' + str(i+1), tf.keras.layers.BatchNormalization())
                self.residual_layers.append('bn' + str(i+1))
        
        else:
            if batch_normalization:
                setattr(self, 'bn' + str(i+1), tf.keras.layers.BatchNormalization())
                self.residual_layers.append('bn' + str(i+1))
            
            setattr(self, 'conv' + str(i+1), tf.keras.layers.Conv2D(filters[i], kernel_size, padding='same'))
            self.residual_layers.append('conv' + str(i+1))

            
            
  def call(self, input_tensor, training=False):
    
    x = input_tensor
    
    for layer in self.residual_layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            x = getattr(self, layer)(x)
        else: 
            x = getattr(self, layer)(x, training=False)
        x = tf.nn.relu(x)
        
    x += input_tensor
    return tf.nn.relu(x)

NUM_CLASSES = 10
INPUT_SHAPE = (32, 32, 3)

# Architecture
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', input_shape=INPUT_SHAPE))

model.add(ResnetIdentityBlock((3,3), filters=(512, 512)))
model.add(tf.keras.layers.Conv2D(256, (3, 3), padding='same'))

model.add(ResnetIdentityBlock((3,3), filters=(256, 256)))
model.add(ResnetIdentityBlock((3,3), filters=(256, 256)))
model.add(ResnetIdentityBlock((3,3), filters=(256, 256)))
model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same'))


model.add(ResnetIdentityBlock((3,3), filters=(128, 128)))
model.add(ResnetIdentityBlock((3,3), filters=(128, 128)))
model.add(ResnetIdentityBlock((3,3), filters=(128, 128)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same'))

model.add(ResnetIdentityBlock((2,2), filters=(64, 64)))
model.add(ResnetIdentityBlock((2,2), filters=(64, 64)))
model.add(ResnetIdentityBlock((2,2), filters=(64, 64)))

model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.AveragePooling2D(pool_size=8))
model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(128))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'))

model.summary()
model.load_weights('./best_model.h5')
