# %%
# Imports
import numpy as np
import tensorflow as tf

# %%
# Show available GPU
tf.config.list_physical_devices('GPU')

# %%
# Define model
class L_net(tf.keras.Model):
    """ Standard model
    """
    def __init__(self, checkpoint=None):
        super().__init__()
        if checkpoint == None:
            self.model = tf.keras.models.Sequential([
                tf.keras.layers.Input(shape=(24,), dtype='float32', name='input_layer'),
                tf.keras.layers.Dense(32, activation='relu', name='hidden_layer'),
                tf.keras.layers.Dropout(0.2, name='dropout_layer'),
                tf.keras.layers.Dense(3, activation='linear', name='output_layer')
            ])
        else:
            self.model = tf.keras.models.load_model(checkpoint)
    
    def call(self, x):
        return self.model(x)

net = L_net()
net

# %%
# Get inputs
inputs = [0.034375000000000044, -0.965625, -0.965625, -0.965625, -0.965625, -0.965625, -0.05102040816326525, -0.08354430379746836, -0.9736842105263158, 0.2533333333333334, 0.4933333333333334, 0.3070175438596492, -0.9746835443037974, -0.9829931972789115, -0.9843505477308294, -0.9812206572769953, -0.9780907668231612, -0.974960876369327, -0.9687010954616588, -0.8622848200312989, 0.011857700679684058, 1.5707958384961647, -0.0014751726864282513, -0.0025529140697340127]
inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
inputs = tf.reshape(inputs, shape=(1, 24))
inputs 

# %%
# Use L_net
outputs = net(inputs).numpy()
outputs