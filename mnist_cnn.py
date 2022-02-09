#!/usr/bin/env python3
import argparse
import datetime
import os
import re

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

# a507688b-17c7-11e8-9de3-00505601122b
# bee39584-17d2-11e8-9de3-00505601122b

import numpy as np
import tensorflow as tf

from mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--cnn", default='C-8-3-5-same,CB-8-3-5-valid,R-[CB-8-3-1-same,CB-8-3-1-same],F,H-50', type=str,
                    help="CNN architecture.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")


# If you add more arguments, ReCodEx will keep them with your default values.

# The neural network model
class Network(tf.keras.Model):
    def __init__(self, args):
        # TODO: Create the model. The template uses functional API, but
        # feel free to use subclassing if you want.
        inputs = tf.keras.layers.Input(shape=[MNIST.H, MNIST.W, MNIST.C])

        # TODO: Add CNN layers specified by `args.cnn`, which contains
        # comma-separated list of the following layers:
        # - `C-filters-kernel_size-stride-padding`: Add a convolutional layer with ReLU
        #   activation and specified number of filters, kernel size, stride and padding.
        # - `CB-filters-kernel_size-stride-padding`: Same as `C`, but use batch normalization.
        #   In detail, start with a convolutional layer without bias and activation,
        #   then add batch normalization layer, and finally ReLU activation.
        # - `M-pool_size-stride`: Add max pooling with specified size and stride, using
        #   the default "valid" padding.
        # - `R-[layers]`: Add a residual connection. The `layers` contain a specification
        #   of at least one convolutional layer (but not a recursive residual connection `R`).
        #   The input to the `R` layer should be processed sequentially by `layers`, and the
        #   produced output (after the ReLU nonlinearty of the last layer) should be added
        #   to the input (of this `R` layer).
        # - `F`: Flatten inputs. Must appear exactly once in the architecture.
        # - `H-hidden_layer_size`: Add a dense layer with ReLU activation and specified size.
        # - `D-dropout_rate`: Apply dropout with the given dropout rate.
        # You can assume the resulting network is valid; it is fine to crash if it is not.
        #
        # Produce the results in variable `hidden`.
        def create_layer(input, layer):
            if layer[:2] == 'C-':
                arguments = layer.split('-')
                if arguments[4][-1] == ']':
                    arguments[4] = arguments[4][:-1]
                return tf.keras.layers.Conv2D(filters=int(arguments[1]), kernel_size=int(arguments[2]),
                                                strides=(int(arguments[3]), int(arguments[3])), padding=arguments[4],
                                                activation=tf.keras.activations.relu)(input)

            elif layer[:2] == 'CB':
                arguments = layer.split('-')
                if arguments[4][-1] == ']':
                    arguments[4] = arguments[4][:-1]
                a = tf.keras.layers.Conv2D(filters=int(arguments[1]), kernel_size=int(arguments[2]),
                                                strides=(int(arguments[3]), int(arguments[3])), padding=arguments[4],
                                                activation=None, use_bias=False)(input)
                b = tf.keras.layers.BatchNormalization()(a)
                return tf.keras.activations.relu(b)

            elif layer[0] == "M":
                arguments = layer.split("-")
                return tf.keras.layers.MaxPool2D(pool_size=(int(arguments[1]), int(arguments[1])), strides=int(arguments[2]))(input)

            elif layer[0] == "F":
                return tf.keras.layers.Flatten()(input)

            elif layer[0] == "H":
                arguments = layer.split("-")
                return tf.keras.layers.Dense(units=int(arguments[1]), activation=tf.keras.activations.relu)(input)

            elif layer[0] == "D":
                arguments = layer.split("-")
                return tf.keras.layers.Dropout(rate=float(arguments[1]))(input)

        layers_list = re.split(',', args.cnn)

        residual = False
        hidden = inputs
        for layer in layers_list:
            if layer[0] != 'R' and not residual:
                hidden = create_layer(input=hidden, layer=layer)
            elif layer[0] == 'R' and not residual:
                from_where = hidden
                hidden = create_layer(input=hidden, layer=layer[3:])
                residual = True
            elif residual:
                hidden = create_layer(input=hidden, layer=layer)
                hidden = tf.keras.layers.Add()([hidden, from_where])
                residual = False

        #
        # hidden = ...
        #
        # # Add the final output layer
        outputs = tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax)(hidden)

        super().__init__(inputs=inputs, outputs=outputs)
        self.compile(
            optimizer=tf.optimizers.Adam(),
            loss=tf.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100,
                                                          profile_batch=0)
        self.tb_callback._close_writers = lambda: None  # A hack allowing to keep the writers open.


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

    # Load the data
    mnist = MNIST()

    # Create the network and train
    network = Network(args)
    network.fit(
        mnist.train.data["images"], mnist.train.data["labels"],
        batch_size=args.batch_size, epochs=args.epochs,
        validation_data=(mnist.dev.data["images"], mnist.dev.data["labels"]),
        callbacks=[network.tb_callback],
    )

    # Compute test set accuracy and return it
    test_logs = network.evaluate(
        mnist.test.data["images"], mnist.test.data["labels"], batch_size=args.batch_size, return_dict=True,
    )
    network.tb_callback.on_epoch_end(args.epochs, {"val_test_" + metric: value for metric, value in test_logs.items()})

    return test_logs["accuracy"]


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)