import tensorflow as tf
from keras.utils import plot_model
from keras.layers import Input, Dense, Conv1D, Conv2D, Flatten, concatenate
from keras.models import Model
from keras.optimizers import SGD
import config

def softmax_cross_entropy_with_logits(y_true, y_pred):
    p = y_pred
    pi = y_true

    zero = tf.zeros(shape=tf.shape(pi), dtype=tf.float32)
    where = tf.equal(pi, zero)

    negatives = tf.fill(tf.shape(pi), -100.0)
    p = tf.where(where, negatives, p)
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=pi, logits=p)

    return loss

class nn_model():
    def __init__(self):
        # This returns a tensor
        input1 = Input(shape=(3,5,5))
        input2 = Input(shape=(3,5,5))
        input3 = Input(shape=(3,5,5))
        input4 = Input(shape=(3,5,5))

        # a layer instance is callable on a tensor, and returns a tensor
        conv = Conv2D(1, kernel_size = (3,5), activation='linear')
        card1 = conv(input1)
        card2 = conv(input2)
        card3 = conv(input3)
        card4 = conv(input4)

        merged = concatenate([card1, card2, card3, card4], axis=-1)
        predictions = Dense(20, activation='softmax',
                            name='policy_head')(merged)
        value = Dense(1, activation='softmax',
                      name='value_head')(merged)
        # This creates a model that includes
        # the Input layer and three Dense layers

        model = Model(inputs=[input1, input2, input3, input4], outputs=[predictions, value])
        model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        self.model = model


