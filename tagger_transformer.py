#!/usr/bin/env python3
# d4215ecb-c593-11e8-a4be-00505601122b
# c751264b-78ee-11eb-a1a9-005056ad4f31
# bee39584-17d2-11e8-9de3-00505601122b
import argparse
import datetime
import os
import re

from tensorflow.python.ops.init_ops_v2 import GlorotUniform
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import numpy as np
import tensorflow as tf

from morpho_dataset import MorphoDataset

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--max_sentences", default=None, type=int, help="Maximum number of sentences to load.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--transformer_dropout", default=0., type=float, help="Transformer dropout.")
parser.add_argument("--transformer_expansion", default=4, type=float, help="Transformer FFN expansion factor.")
parser.add_argument("--transformer_heads", default=4, type=int, help="Transformer heads.")
parser.add_argument("--transformer_layers", default=2, type=int, help="Transformer layers.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=None, type=int, help="Maximum number of threads to use.")
parser.add_argument("--we_dim", default=64, type=int, help="Word embedding dimension.")
# If you add more arguments, ReCodEx will keep them with your default values.

class Network(tf.keras.Model):
    class FFN(tf.keras.layers.Layer):
        def __init__(self, dim, expansion, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.dim, self.expansion = dim, expansion
            # TODO: Create the required layers -- first a ReLU-activated dense
            # layer with `dim * expansion` units, followed by a dense layer
            # with `dim` units without an activation.
            self.layers = []
            self.layers.append(tf.keras.layers.Dense(dim * expansion, activation=tf.nn.relu))
            self.layers.append(tf.keras.layers.Dense(dim))

        def get_config(self):
            return {"dim": self.dim, "expansion": self.expansion}

        def call(self, inputs):
            # TODO: Execute the FFN Transformer layer.
            x = inputs
            for layer in self.layers:
                x = layer(x)
            return x

    class SelfAttention(tf.keras.layers.Layer):
        def __init__(self, dim, heads, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.dim, self.heads = dim, heads
            # TODO: Create weight matrices W_Q, W_K, W_V and W_O using `self.add_weight`,
            # each with shape `[dim, dim]`; otherwise, default arguments should be used,
            # which mean trainable float32 matrices initialized with "glorot_uniform".
            self.W_Q = self.add_weight('W_Q', shape=[dim, dim])
            self.W_K = self.add_weight('W_K', shape=[dim, dim])
            self.W_V = self.add_weight('W_V', shape=[dim, dim])
            self.W_O = self.add_weight('W_O', shape=[dim, dim])

        def get_config(self):
            return {"dim": self.dim, "heads": self.heads}

        def matrixOperation(self, inputs, batch_size, max_sentence_len, W_M):
            M = inputs @ W_M
            M = tf.reshape(M, [batch_size, max_sentence_len, self.heads, self.dim // self.heads])
            M = tf.transpose(M, [0, 2, 1, 3])
            return M

        def call(self, inputs, mask):
            # TODO: Execute the self-attention layer.
            #
            # Start by computing Q, K and V. In all cases:
            # - first multiply `inputs` by the corresponding weight matrix W_Q/W_K/W_V,
            # - reshape via `tf.reshape` to [batch_size, max_sentence_len, heads, dim // heads],
            # - transpose via `tf.transpose` to [batch_size, heads, max_sentence_len, dim // heads].
            batch_size = tf.shape(inputs)[0]
            max_sentence_len = tf.shape(inputs)[1]
            Q = self.matrixOperation(inputs, batch_size, max_sentence_len, self.W_Q)
            K = self.matrixOperation(inputs, batch_size, max_sentence_len, self.W_K)
            V = self.matrixOperation(inputs, batch_size, max_sentence_len, self.W_V)
            
            # TODO: Continue by computing the self-attention weights as Q @ K^T,
            # normalizing by the square root of `dim // heads`.
            weights = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(tf.cast(self.dim // self.heads, tf.float32))

            # TODO: Apply the softmax, but including a suitable mask, which ignores all padding words.
            # The original `mask` is a bool matrix of shape [batch_size, max_sentence_len]
            # indicating which words are valid (True) or padding (False).
            # - You can perform the masking manually, by setting the attention weights
            #   of padding words to -1e9.
            # - Alternatively, you can use the fact that tf.keras.layers.Softmax accepts a named
            #   boolean argument `mask` indicating the valid (True) or padding (False) elements.
            m = tf.cast(mask, tf.int32)
            m = tf.einsum('bi,bj->bij', m, m)
            m = tf.expand_dims(m, 1)
            m = tf.tile(m, [1, self.heads, 1, 1])

            n = tf.cast(1 - m, tf.float32) * (-1e9)
            m = tf.cast(m, tf.float32)

            weights = tf.nn.softmax(weights * m + n)

            # TODO: Finally,
            # - take a weighted combination of values V according to the computed attention,
            # - transpose the result to [batch_size, max_sentence_len, heads, dim // heads],
            # - reshape to [batch_size, max_sentence_len, dim],
            # - multiply the result by the W_O matrix.
            attention = weights @ V
            attention = tf.transpose(attention, [0, 2, 1, 3])
            attention = tf.reshape(attention, [batch_size, max_sentence_len, self.dim])

            return attention @ self.W_O

    class Transformer(tf.keras.layers.Layer):
        def __init__(self, layers, dim, expansion, heads, dropout, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.layers, self.dim, self.expansion, self.heads, self.dropout = layers, dim, expansion, heads, dropout
            # TODO: Create the required number of transformer layers, each consisting of
            # - a self-attention layer followed by a dropout layer and layer normalization,
            # - a FFN layer followed by a dropout layer and layer normalization.
            self.subLayers = []
            for i in range(layers):
                subLayer = []
                subLayer.append(Network.SelfAttention(dim, heads))
                subLayer.append(tf.keras.layers.Dropout(dropout))
                subLayer.append(tf.keras.layers.LayerNormalization())
                subLayer.append(Network.FFN(dim, expansion))
                subLayer.append(tf.keras.layers.Dropout(dropout))
                subLayer.append(tf.keras.layers.LayerNormalization())
                self.subLayers.append(subLayer)

        def get_config(self):
            return {name: getattr(self, name) for name in ["layers", "dim", "expansion", "heads", "dropout"]}

        def call(self, inputs, mask):
            # TODO: Start by computing the sinusoidal positional embeddings.
            # They have a shape `[max_sentence_len, dim]` and
            # - for `i < dim / 2`, the value on index `[pos, i]` should be
            #     `sin(pos / 10000 ** (2 * i / dim))`
            # - the value on index `[pos, dim / 2 + i]` should be
            #     `cos(pos / 10000 ** (2 * i / dim))`
            # This order is the same as in the visualization on the slides, but
            # different from the original paper where `sin` and `cos` interleave.
            max_sentence_len = tf.shape(inputs)[1]
            dim = self.dim

            pos_1, i_1 = tf.meshgrid(range(max_sentence_len), range(dim // 2), indexing='ij')
            pos_2, i_2 = tf.meshgrid(range(max_sentence_len), range(dim // 2), indexing='ij')

            pos_1 = tf.cast(pos_1, tf.float32)
            pos_2 = tf.cast(pos_2, tf.float32)
            i_1 = tf.cast(i_1, tf.float32)
            i_2 = tf.cast(i_2, tf.float32)

            embeddings_1 = tf.sin(pos_1 / (10000 ** (2 * i_1 / dim)))
            embeddings_2 = tf.cos(pos_2 / (10000 ** (2 * i_2 / dim)))
            embeddings = tf.concat([embeddings_1, embeddings_2], axis=1)

            # TODO: Add the positional embeddings to the `inputs` and then
            # perform the given number of transformer layer, composed of
            # - a self-attention sub-layer, followed by
            # - a FFN sub-layer.
            # In each sub-layer, compute the corresponding operation followed
            # by dropout, add the original sub-layer input and pass the result
            # through LayerNorm. Note that the given `mask` should be passed
            # to the self-attention operation to ignore the padding words.
            x = inputs + embeddings

            for subLayer in self.subLayers:
                input_0 = x
                x = subLayer[0](x, mask=mask)
                x = subLayer[1](x)
                x = x + input_0
                x = subLayer[2](x)

                input_1 = x
                x = subLayer[3](x)
                x = subLayer[4](x)
                x = x + input_1
                x = subLayer[5](x)

            return x

    def __init__(self, args, train):
        # Implement a transformer encoder network. The input `words` is
        # a RaggedTensor of strings, each batch example being a list of words.
        words = tf.keras.layers.Input(shape=[None], dtype=tf.string, ragged=True)

        # TODO(tagger_we): Map strings in `words` to indices by using the `word_mapping` of `train.forms`.
        x = train.forms.word_mapping(words)

        # TODO(tagger_we): Embed input words with dimensionality `args.we_dim`. Note that the `word_mapping`
        # provides a `vocab_size()` call returning the number of unique words in the mapping.
        numberOfUniqueForms = train.forms.word_mapping.vocab_size()
        x = tf.keras.layers.Embedding(input_dim=numberOfUniqueForms, output_dim=args.we_dim)(x)

        # TODO: Call the Transformer layer:
        # - create a `Network.Transformer` layer, using suitable options from `args`
        #   (using `args.we_dim` for the `dim` argument),
        # - when calling the layer, convert the ragged tensor with the input words embedding
        #   to a dense one, and also pass the following argument as a mask:
        #     `mask=tf.sequence_mask(ragged_tensor_with_input_words_embeddings.row_lengths())`
        # - finally, convert the result back to a ragged tensor.
        transformerLayer = Network.Transformer(
            layers=args.transformer_layers,
            dim=args.we_dim,
            expansion=args.transformer_expansion,
            heads=args.transformer_heads,
            dropout=args.transformer_dropout
        )

        x_dense = x.to_tensor()
        x_dense = transformerLayer(x_dense, mask=tf.sequence_mask(x.row_lengths()))
        x = tf.RaggedTensor.from_tensor(x_dense, x.row_lengths())

        # TODO(tagge_we): Add a softmax classification layer into as many classes as there are unique
        # tags in the `word_mapping` of `train.tags`. However, because we are applying the
        # the Dense layer to a ragged tensor, we need to wrap the Dense layer in
        # a tf.keras.layers.TimeDistributed.
        numberOfUniqueTags = train.tags.word_mapping.vocab_size()
        predictions = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(numberOfUniqueTags, activation=tf.nn.softmax))(x)

        # Check that the created predictions are a 3D tensor.
        assert predictions.shape.rank == 3
        super().__init__(inputs=words, outputs=predictions)
        self.compile(optimizer=tf.optimizers.Adam(),
                     loss=tf.losses.SparseCategoricalCrossentropy(),
                     metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")])

        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, update_freq=100, profile_batch=0)
        self.tb_callback._close_writers = lambda: None # A hack allowing to keep the writers open.

    # Note that in TF 2.4, computing losses and metrics on RaggedTensors is not yet
    # supported (it will be in TF 2.5). Therefore, we override the `train_step` method
    # to support it, passing the "flattened" predictions and gold data to the loss
    # and metrics.
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            # Check that both the gold data and predictions are RaggedTensors.
            assert isinstance(y_pred, tf.RaggedTensor) and isinstance(y, tf.RaggedTensor)
            loss = self.compiled_loss(y.values, y_pred.values, regularization_losses=self.losses)
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.compiled_metrics.update_state(y.values, y_pred.values)
        return {m.name: m.result() for m in self.metrics}

    # Analogously to `train_step`, we also need to override `test_step`.
    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        loss = self.compiled_loss(y.values, y_pred.values, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y.values, y_pred.values)
        return {m.name: m.result() for m in self.metrics}

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
    morpho = MorphoDataset("czech_cac", max_sentences=args.max_sentences)

    # Create the network and train
    network = Network(args, morpho.train)

    # TODO(tagger_we): Construct dataset for training, which should contain pairs of
    # - tensor of string words (forms) as input
    # - tensor of integral tag ids as targets.
    # To create the identifiers, use the `word_mapping` of `morpho.train.tags`.
    def tagging_dataset(forms, lemmas, tags):
        return forms, morpho.train.tags.word_mapping(tags)

    def create_dataset(name):
        dataset = getattr(morpho, name).dataset
        dataset = dataset.map(tagging_dataset)
        dataset = dataset.shuffle(len(dataset), seed=args.seed) if name == "train" else dataset
        dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
        return dataset
    train, dev, test = create_dataset("train"), create_dataset("dev"), create_dataset("test")

    network.fit(train, epochs=args.epochs, validation_data=dev, callbacks=[network.tb_callback])

    test_logs = network.evaluate(test, return_dict=True)
    network.tb_callback.on_epoch_end(args.epochs, {"val_test_" + metric: value for metric, value in test_logs.items()})

    # Return test set accuracy for ReCodEx to validate
    return test_logs["accuracy"]

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
