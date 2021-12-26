import tensorflow as tf
import mitdeeplearning as mdl
import numpy as np
import matplotlib.pyplot as plt

sport = tf.constant("Tennis", tf.string)
number = tf.constant(1.41421356237, tf.float64)

print(sport)
print("`number` is a {}-d Tensor".format(tf.rank(number).numpy()))


class OurDenseLayer(tf.keras.layers.Layer):
    def __init__(self, n_output_nodes):
        super(OurDenseLayer, self).__init__()
        self.n_output_nodes = n_output_nodes

    def build(self, input_shape):
        d = int(input_shape[-1])
        # Define and initialize parameters: a weight matrix W and bias b
        # Note that parameter initialization is random!
        self.W = self.add_weight("weight", shape=[d, self.n_output_nodes])  # note the dimensionality
        self.b = self.add_weight("bias", shape=[1, self.n_output_nodes])  # note the dimensionality

    def call(self, x):
        '''TODO: define the operation for z (hint: use tf.matmul)'''
        z = tf.matmul(x, self.W) + self.b
        y = tf.sigmoid(z)
        return y


class SubclassModel(tf.keras.Model):

    # In __init__, we define the Model's layers
    def __init__(self, n_output_nodes):
        super(SubclassModel, self).__init__()
        '''TODO: Our model consists of a single Dense layer. Define this layer.'''
        self.dense_layer = OurDenseLayer(n_output_nodes)

    # In the call function, we define the Model's forward pass.
    def call(self, inputs):
        return self.dense_layer(inputs)


n_output_nodes = 3
model = SubclassModel(n_output_nodes)

x_input = tf.constant([[1, 2.]], shape=(1, 2))

print(model.call(x_input))
