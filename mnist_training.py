#!/usr/bin/env python3
# Solved in team:
# a507688b-17c7-11e8-9de3-00505601122b
# bee39584-17d2-11e8-9de3-00505601122b

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
parser.add_argument("--decay", default=None, type=str, help="Learning decay rate type")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layer", default=200, type=int, help="Size of the hidden layer.")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Initial learning rate.")
parser.add_argument("--learning_rate_final", default=None, type=float, help="Final learning rate.")
parser.add_argument("--momentum", default=None, type=float, help="Momentum.")
parser.add_argument("--optimizer", default="SGD", type=str, help="Optimizer to use.")
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
    if args.recodex:
        tf.keras.utils.get_custom_objects()["glorot_uniform"] = tf.initializers.GlorotUniform(seed=args.seed)
        tf.keras.utils.get_custom_objects()["orthogonal"] = tf.initializers.Orthogonal(seed=args.seed)
        tf.keras.utils.get_custom_objects()["uniform"] = tf.initializers.RandomUniform(seed=args.seed)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load data
    mnist = MNIST()

    # Create the model
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=[MNIST.H, MNIST.W, MNIST.C]),
        tf.keras.layers.Dense(args.hidden_layer, activation=tf.nn.relu),
        tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax),
    ])

    decay_steps = mnist.train.size / args.batch_size

    if args.optimizer == "SGD":
        if args.momentum == None:
            optimizer = tf.optimizers.SGD(learning_rate=args.learning_rate)
        else:
            optimizer = tf.optimizers.SGD(learning_rate=args.learning_rate, momentum=args.momentum)
    else:  # ADAM
        if args.decay == None:
            optimizer = tf.optimizers.Adam(learning_rate=args.learning_rate)
        elif args.decay == "polynomial":
            optimizer = tf.optimizers.Adam(learning_rate=tf.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=args.learning_rate, end_learning_rate=args.learning_rate_final,
                decay_steps=decay_steps
            ))
        else:  # exponential
            optimizer = tf.optimizers.Adam(learning_rate=tf.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=args.learning_rate, decay_steps=decay_steps,
                decay_rate=tf.math.pow(args.learning_rate_final / args.learning_rate, 1 / args.epochs)
            ))
    model.compile(
        optimizer=optimizer,
        loss=tf.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.metrics.SparseCategoricalAccuracy("accuracy")],
    )

    tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0)
    tb_callback._close_writers = lambda: None  # Ugly hack allowing to log also test data metrics.
    model.fit(
        mnist.train.data["images"], mnist.train.data["labels"],
        batch_size=args.batch_size, epochs=args.epochs,
        validation_data=(mnist.dev.data["images"], mnist.dev.data["labels"]),
        callbacks=[tb_callback],
    )

    test_logs = model.evaluate(
        mnist.test.data["images"], mnist.test.data["labels"], batch_size=args.batch_size, return_dict=True,
    )
    tb_callback.on_epoch_end(args.epochs, {"val_test_" + metric: value for metric, value in test_logs.items()})

    # Return test accuracy for ReCodEx to validate
    return test_logs["accuracy"]


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
