#!/usr/bin/env python3
# Solved in team:
# a507688b-17c7-11e8-9de3-00505601122b
# bee39584-17d2-11e8-9de3-00505601122b
# 1af6e984-1812-11e8-9de3-00505601122b

import argparse
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layers", default=[100], nargs="*", type=int, help="Hidden layer sizes.")
parser.add_argument("--models", default=3, type=int, help="Number of models.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")


# If you add more arguments, ReCodEx will keep them with your default values.

def main(args):
    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Load data
    mnist = MNIST()

    # Create models
    models = []

    for model in range(args.models):
        if args.recodex:
            tf.keras.utils.get_custom_objects()["glorot_uniform"] = tf.initializers.GlorotUniform(
                seed=args.seed + model)
            tf.keras.utils.get_custom_objects()["orthogonal"] = tf.initializers.Orthogonal(seed=args.seed + model)
            tf.keras.utils.get_custom_objects()["uniform"] = tf.initializers.RandomUniform(seed=args.seed + model)

        models.append(tf.keras.Sequential([
                                              tf.keras.layers.Flatten(input_shape=[MNIST.H, MNIST.W, MNIST.C]),
                                          ] + [tf.keras.layers.Dense(hidden_layer, activation=tf.nn.relu) for
                                               hidden_layer in args.hidden_layers] + [
                                              tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax),
                                          ]))

        models[-1].compile(
            optimizer=tf.optimizers.Adam(),
            loss=tf.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )

        print("Training model {}: ".format(model + 1), end="", file=sys.stderr, flush=True)
        models[-1].fit(
            mnist.train.data["images"], mnist.train.data["labels"],
            batch_size=args.batch_size, epochs=args.epochs, verbose=0
        )
        print("Done", file=sys.stderr)

    individual_accuracies, ensemble_accuracies = [], []
    for model_num in range(args.models):
        # Compute the accuracy on the dev set for the individual `models[model]`.
        individual_accuracy = \
            (models[model_num].evaluate(mnist.dev.data["images"], mnist.dev.data["labels"],
                                        batch_size=args.batch_size))[1]

        # Compute the accuracy on the dev set for the ensemble `models[0:model+1].
        if model_num == 0:
            individual_accuracies.append(individual_accuracy)
            ensemble_accuracies.append(individual_accuracy)
            continue

        # Create model
        input_layer = tf.keras.layers.Input([MNIST.H, MNIST.W, MNIST.C])

        ensemble_models = []
        for i in range(model_num + 1):
            x = models[i](input_layer)
            ensemble_models.append(x)

        layer_average = tf.keras.layers.Average()
        output_layer = layer_average(ensemble_models)

        ensemble_model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        ensemble_model.compile(metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")])

        ensemble_accuracy = \
            ensemble_model.evaluate(mnist.dev.data["images"], mnist.dev.data["labels"], batch_size=args.batch_size)[1]

        # Store the accuracies
        individual_accuracies.append(individual_accuracy)
        ensemble_accuracies.append(ensemble_accuracy)
    return individual_accuracies, ensemble_accuracies


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    individual_accuracies, ensemble_accuracies = main(args)
    for model, (individual_accuracy, ensemble_accuracy) in enumerate(zip(individual_accuracies, ensemble_accuracies)):
        print("Model {}, individual accuracy {:.2f}, ensemble accuracy {:.2f}".format(
            model + 1, 100 * individual_accuracy, 100 * ensemble_accuracy))
