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
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")


# If you add more arguments, ReCodEx will keep them with your default values.

# The neural network model
class Network(tf.keras.Model):
    def __init__(self, args):
        # Create a model with two inputs, both images of size [MNIST.H, MNIST.W, MNIST.C].
        images = (
            tf.keras.layers.Input(shape=[MNIST.H, MNIST.W, MNIST.C]),
            tf.keras.layers.Input(shape=[MNIST.H, MNIST.W, MNIST.C]),
        )

        # TODO: The model starts by passing each input image through the same
        # subnetwork (with shared weights), which should perform
        # - convolution with 10 filters, 3x3 kernel size, stride 2, "valid" padding, ReLU activation,
        # - convolution with 20 filters, 3x3 kernel size, stride 2, "valid" padding, ReLU activation,
        # - flattening layer,
        # - fully connected layer with 200 neurons and ReLU activation,
        # obtaining a 200-dimensional feature representation of each image.

        # Create the shared subnetwork
        input_shared_subnetwork = tf.keras.layers.Input(shape=[MNIST.H, MNIST.W, MNIST.C])
        convolution_layer_1 = tf.keras.layers.Conv2D(kernel_size=[3, 3], filters=10, strides=2, padding="valid",
                                                     activation=tf.nn.relu)(input_shared_subnetwork)
        convolution_layer_2 = tf.keras.layers.Conv2D(kernel_size=[3, 3], filters=20, strides=2, padding="valid",
                                                     activation=tf.nn.relu)(convolution_layer_1)
        flattened_layer = tf.keras.layers.Flatten()(convolution_layer_2)
        connected_layer = tf.keras.layers.Dense(200, activation=tf.nn.relu)(flattened_layer)
        output_shared_subnetwork = connected_layer

        shared_subnetwork = tf.keras.models.Model(input_shared_subnetwork, output_shared_subnetwork)

        # run two input images trough the subnetwork
        represent_image1 = shared_subnetwork(images[0])
        represent_image2 = shared_subnetwork(images[1])

        # TODO: Using the computed representations, it should produce four outputs:
        # - first, compute _direct prediction_ whether the first digit is
        #   greater than the second, by
        #   - concatenating the two 200-dimensional image representations,
        #   - processing them using another 200-neuron ReLU dense layer
        #   - computing one output using a dense layer with `tf.nn.sigmoid` activation
        # - then, classify the computed representation of the first image using
        #   a densely connected softmax layer into 10 classes;
        # - then, classify the computed representation of the second image using
        #   the same connected layer (with shared weights) into 10 classes;
        # - finally, compute _indirect prediction_ whether the first digit
        #   is greater than second, by comparing the predictions from the above
        #   two outputs.

        # compute the direct prediction
        concatenated_layer = tf.keras.layers.concatenate([represent_image1, represent_image2])
        dense_layer_1 = tf.keras.layers.Dense(200, activation=tf.nn.relu)(concatenated_layer)

        direct_prediction = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(dense_layer_1)

        # create again a shared subnetwork to classify digits
        input_digit_classification = tf.keras.Input(shape=200)
        dense_layer_2 = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(input_digit_classification)
        output_digit_classification = dense_layer_2

        digit_classification = tf.keras.models.Model(input_digit_classification, output_digit_classification)

        # receive two digits and compute indirect prediction
        digit_1 = digit_classification(represent_image1)
        digit_2 = digit_classification(represent_image2)

        indirect_prediction = tf.argmax(digit_1, -1) > tf.argmax(digit_2, -1)

        outputs = {
            "direct_prediction": direct_prediction,
            "digit_1": digit_1,
            "digit_2": digit_2,
            "indirect_prediction": indirect_prediction,
        }

        # Finally, construct the model.
        super().__init__(inputs=images, outputs=outputs)

        # Note that for historical reasons, names of a functional model outputs
        # (used for displayed losses/metric names) are derived from the name of
        # the last layer of the corresponding output. Here we instead use
        # the keys of the `outputs` dictionary.
        self.output_names = sorted(outputs.keys())

        # TODO: Train the model by computing appropriate losses of
        # direct_prediction, digit_1, digit_2. Regarding metrics, compute
        # the accuracy of both the direct and indirect predictions; name both
        # metrics "accuracy" (i.e., pass "accuracy" as the first argument of
        # the metric object).
        self.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss={
                "direct_prediction": tf.losses.BinaryCrossentropy(),
                "digit_1": tf.losses.SparseCategoricalCrossentropy(),
                "digit_2": tf.losses.SparseCategoricalCrossentropy(),
            },
            metrics={
                "direct_prediction": [tf.metrics.BinaryAccuracy(name="accuracy")],
                "indirect_prediction": [tf.metrics.BinaryAccuracy(name="accuracy")],
            }
        )

        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100,
                                                          profile_batch=0)
        self.tb_callback._close_writers = lambda: None  # A hack allowing to keep the writers open.

    # Create an appropriate dataset using the MNIST data.
    def create_dataset(self, mnist_dataset, args, training=False):
        # Start by using the original MNIST data
        dataset = tf.data.Dataset.from_tensor_slices((mnist_dataset.data["images"], mnist_dataset.data["labels"]))

        # TODO: If `training`, shuffle the data with `buffer_size=10000` and `seed=args.seed`
        if training:
            dataset = dataset.shuffle(buffer_size=10000, seed=args.seed)

        # TODO: Combine pairs of examples by creating batches of size 2
        dataset = dataset.batch(batch_size=2)

        # TODO: Map pairs of images to elements suitable for our model. Notably,
        # the elements should be pairs `(input, output)`, with
        # - `input` being a pair of images,
        # - `output` being a dictionary with keys digit_1, digit_2, direct_prediction
        #   and indirect_prediction.
        def create_element(images, labels):
            input = (images[0], images[1])
            output = {"digit_1": labels[0],
                      "digit_2": labels[1],
                      'direct_prediction': labels[0] > labels[1],
                      'indirect_prediction': labels[0] > labels[1]
                      }
            return input, output

        dataset = dataset.map(create_element)

        # TODO: Create batches of size `args.batch_size`
        dataset = dataset.batch(batch_size=args.batch_size)

        return dataset


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

    # Create the network
    network = Network(args)

    # Construct suitable datasets from the MNIST data.
    train = network.create_dataset(mnist.train, args, training=True)
    dev = network.create_dataset(mnist.dev, args)
    test = network.create_dataset(mnist.test, args)

    # Train
    network.fit(train, epochs=args.epochs, validation_data=dev, callbacks=[network.tb_callback])

    # Compute test set metrics and return them
    test_logs = network.evaluate(test, return_dict=True)
    network.tb_callback.on_epoch_end(args.epochs, {"val_test_" + metric: value for metric, value in test_logs.items()})

    return test_logs


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
