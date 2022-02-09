#!/usr/bin/env python3
import argparse
import datetime
import os
import re
# Solved in team:
# a507688b-17c7-11e8-9de3-00505601122b
# bee39584-17d2-11e8-9de3-00505601122b

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--cnn", default="5-3-2,10-3-2", type=str, help="CNN architecture.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")


# If you add more arguments, ReCodEx will keep them with your default values.

class Convolution:
    def __init__(self, channels, kernel_size, stride, input_shape):
        # Create convolutional layer with the given arguments
        # and given input shape (e.g., [28, 28, 1]).
        self._channels = channels
        self._kernel_size = kernel_size
        self._stride = stride

        # Here the kernel and bias variables are created
        self._kernel = tf.Variable(
            tf.initializers.GlorotUniform(seed=42)(
                [self._kernel_size, self._kernel_size, input_shape[2], self._channels]),
            trainable=True)
        self._bias = tf.Variable(tf.initializers.Zeros()([self._channels]), trainable=True)

    def forward(self, inputs):
        batch_num = inputs.shape[0]
        input_w = inputs.shape[1]

        input_w_cut = input_w - (input_w - self._kernel_size) % self._stride

        # Compute the dimensions of the CONV output volume
        output_w = int((input_w_cut - self._kernel_size) / self._stride) + 1

        # Initialise output of convolution
        conv_output = np.zeros([batch_num, output_w, output_w, self._channels])

        # Iterate through kernel size
        for m in range(self._kernel_size):
            for n in range(self._kernel_size):
                maximum_m = input_w_cut - self._kernel_size + m + 1
                maximum_n = input_w_cut - self._kernel_size + n + 1
                input_cut = inputs[:, m: maximum_m:self._stride, n:maximum_n:self._stride, :]

                conv_output += np.einsum('aijd,de->aije', input_cut, self._kernel[m, n, :, :])

        bias_blank = np.ones(conv_output.shape)
        bias_array = np.einsum('aijd,d->aijd', bias_blank, self._bias)

        output_forward_pass = tf.nn.relu(conv_output + bias_array)

        return output_forward_pass

    def backward(self, inputs, outputs, outputs_gradient):
        input_w = inputs.shape[1]
        input_w_modified = input_w - (input_w - self._kernel_size) % self._stride

        gradient_inputs = np.zeros(inputs.shape)
        gradient_kernel = np.zeros(self._kernel.shape)

        gradient_relu = np.sign(outputs)
        gradient_relu_outputs = np.einsum('aijb,aijb->aijb', gradient_relu, outputs_gradient)

        for m in range(self._kernel_size):
            for n in range(self._kernel_size):
                maximum_m = input_w_modified - self._kernel_size + m + 1
                maximum_n = input_w_modified - self._kernel_size + n + 1

                input_cut = inputs[:, m: maximum_m: self._stride, n:maximum_n:self._stride, :]

                gradient_inputs[:, m: maximum_m: self._stride, n:maximum_n:self._stride, :] += np.einsum(
                    'aije,de->aijd', gradient_relu_outputs, self._kernel[m, n, :, :])

                gradient_kernel[m, n, :, :] = np.einsum('aije,aijd->de', gradient_relu_outputs, input_cut)

        gradient_bias = np.einsum('aije->e', gradient_relu_outputs)

        return [gradient_inputs, [self._kernel, self._bias], [gradient_kernel, gradient_bias]]


class Network:
    def __init__(self, args):
        self._args = args

        # Create the convolutional layers according to `args.cnn`.
        input_shape = [MNIST.H, MNIST.W, MNIST.C]
        self._convolutions = []
        for layer in args.cnn.split(","):
            channels, kernel_size, stride = map(int, layer.split("-"))
            self._convolutions.append(Convolution(channels, kernel_size, stride, input_shape))
            input_shape = [(input_shape[0] - kernel_size) // stride + 1,
                           (input_shape[1] - kernel_size) // stride + 1, channels]

        # Create the classification head
        self._flatten = tf.keras.layers.Flatten(input_shape=input_shape)
        self._classifier = tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax)

        # Create the loss, metric and the optimizer
        self._loss = tf.losses.SparseCategoricalCrossentropy()
        self._accuracy = tf.metrics.SparseCategoricalAccuracy()
        self._optimizer = tf.optimizers.Adam(args.learning_rate)

    def train_epoch(self, dataset):
        for batch in dataset.batches(self._args.batch_size):
            # Forward pass through the convolutions
            hidden = tf.constant(batch["images"])
            convolution_values = [hidden]
            for convolution in self._convolutions:
                hidden = convolution.forward(hidden)
                convolution_values.append(hidden)

            # Run the classification head and compute its gradient
            with tf.GradientTape() as tape:
                tape.watch(hidden)

                predictions = self._flatten(hidden)
                predictions = self._classifier(predictions)
                loss = self._loss(batch["labels"], predictions)

            variables = self._classifier.trainable_variables
            hidden_gradient, *gradients = tape.gradient(loss, [hidden] + variables)

            # Backpropagate the gradient throug the convolutions
            for convolution, inputs, outputs in reversed(
                    list(zip(self._convolutions, convolution_values[:-1], convolution_values[1:]))):
                hidden_gradient, convolution_variables, convolution_gradients = convolution.backward(inputs,
                                                                                                     outputs,
                                                                                                     hidden_gradient)
                variables.extend(convolution_variables)
                gradients.extend(convolution_gradients)

            # Update the weights
            self._optimizer.apply_gradients(zip(gradients, variables))

    def evaluate(self, dataset):
        self._accuracy.reset_states()
        for batch in dataset.batches(self._args.batch_size):
            hidden = batch["images"]
            for convolution in self._convolutions:
                hidden = convolution.forward(hidden)
            hidden = self._flatten(hidden)
            predictions = self._classifier(hidden)
            self._accuracy(batch["labels"], predictions)
        return self._accuracy.result().numpy()


def main(args):
    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    if args.recodex:
        tf.keras.utils.get_custom_objects()["glorot_uniform"] = tf.initializers.GlorotUniform(seed=args.seed)
        tf.keras.utils.get_custom_objects()["orthogonal"] = tf.initializers.Orthogonal(seed=args.seed)
        tf.keras.utils.get_custom_objects()["uniform"] = tf.initializers.RandomUniform(seed=args.seed)

    # Load data, using only 10000 training images
    mnist = MNIST()
    mnist.train._size = 10000

    # Create the model
    network = Network(args)

    for epoch in range(args.epochs):
        network.train_epoch(mnist.train)

        accuracy = network.evaluate(mnist.dev)
        print("Dev accuracy after epoch {} is {:.2f}".format(epoch + 1, 100 * accuracy), flush=True)

    accuracy = network.evaluate(mnist.test)
    print("Test accuracy after epoch {} is {:.2f}".format(epoch + 1, 100 * accuracy), flush=True)

    # Return the test accuracy for ReCodEx to validate.
    return accuracy


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
