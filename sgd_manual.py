#!/usr/bin/env python3
# Solved in team:
# a507688b-17c7-11e8-9de3-00505601122b
# bee39584-17d2-11e8-9de3-00505601122b
# 1af6e984-1812-11e8-9de3-00505601122b

import argparse
import datetime
import os
import re

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layer", default=100, type=int, help="Size of the hidden layer.")
parser.add_argument("--learning_rate", default=0.1, type=float, help="Learning rate.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")


# If you add more arguments, ReCodEx will keep them with your default values.

class Model(tf.Module):
    def __init__(self, args):
        self._args = args
        self._W1 = tf.Variable(
            tf.random.normal([MNIST.W * MNIST.H * MNIST.C, args.hidden_layer], stddev=0.1, seed=args.seed),
            trainable=True)
        self._b1 = tf.Variable(tf.zeros([args.hidden_layer]), trainable=True)
        self._W2 = tf.Variable(
            tf.random.normal([args.hidden_layer, MNIST.LABELS], stddev=0.1, seed=args.seed),
            trainable=True)
        self._b2 = tf.Variable(tf.zeros([MNIST.LABELS]))

    def predict(self, inputs):
        inputs = tf.reshape(inputs, [inputs.shape[0], -1])
        hidden = tf.nn.tanh(inputs @ self._W1 + self._b1)
        probabilities = tf.nn.softmax(hidden @ self._W2 + self._b2)

        return probabilities, hidden, inputs

    def train_epoch(self, dataset):
        for batch in dataset.batches(self._args.batch_size):

            probabilities, hidden, batch_x = self.predict(batch["images"])
            batch_y = tf.one_hot(batch["labels"], MNIST.LABELS)

            # ---------- manual deriv. by B2: -----------
            d_b2 = probabilities - batch_y
            gradient_b2 = tf.math.reduce_mean(d_b2, axis=0)

            # ---------- manual deriv. by W2: -----------
            d_w2 = tf.einsum("ai,aj->aij", hidden, d_b2)
            gradient_w2 = tf.math.reduce_mean(d_w2, axis=0)

            # ---------- manual deriv. by B1: -----------
            tanh = np.ones(hidden.shape) - tf.math.square(hidden)
            d_b1 = tf.math.multiply((d_b2 @ tf.transpose(self._W2)), tanh)
            gradient_b1 = tf.math.reduce_mean(d_b1, axis=0)

            # ---------- manual deriv. by B1: -----------
            d_w1 = tf.einsum("ai,aj->aij", batch_x, d_b1)
            gradient_w1 = tf.math.reduce_mean(d_w1, axis=0)

            gradients = [gradient_w1, gradient_b1, gradient_w2, gradient_b2]
            variables = [self._W1, self._b1, self._W2, self._b2]

            for variable, gradient in zip(variables, gradients):
                variable.assign(variable - self._args.learning_rate * gradient)

    def evaluate(self, dataset):
        # Compute the accuracy of the model prediction
        correct = 0
        for batch in dataset.batches(self._args.batch_size):
            probabilities, _, _ = self.predict(batch["images"])
            correct += np.sum(np.argmax(probabilities, axis=1) == batch["labels"])
        return correct / dataset.size


def main(args):
    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load data
    mnist = MNIST()

    # Create the TensorBoard writer
    writer = tf.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)

    # Create the model
    model = Model(args)

    for epoch in range(args.epochs):
        model.train_epoch(dataset=mnist.train)
        accuracy = model.evaluate(mnist.dev)
        print("Dev accuracy after epoch {} is {:.2f}".format(epoch + 1, 100 * accuracy), flush=True)
        with writer.as_default():
            tf.summary.scalar("dev/accuracy", 100 * accuracy, step=epoch + 1)

    accuracy = model.evaluate(mnist.test)
    print("Test accuracy after epoch {} is {:.2f}".format(epoch + 1, 100 * accuracy), flush=True)
    with writer.as_default():
        tf.summary.scalar("test/accuracy", 100 * accuracy, step=epoch + 1)

    # Return test accuracy for ReCodEx to validate
    return accuracy


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
