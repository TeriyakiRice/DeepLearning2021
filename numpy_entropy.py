# !/usr/bin/env python3
# Solved in team:
# a507688b-17c7-11e8-9de3-00505601122b
# bee39584-17d2-11e8-9de3-00505601122b
# 1af6e984-1812-11e8-9de3-00505601122b
import argparse

import numpy as np

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")


# If you add more arguments, ReCodEx will keep them with your default values.

def main(args):
    dict_data = {}
    with open("numpy_entropy_data.txt", "r") as data:
        for line in data:
            line = line.rstrip("\n")
            dict_data[line] = dict_data.get(line, 0) + 1

    arr_data = np.empty(len(dict_data))
    sum_data = 0
    x = 0

    for key in dict_data:
        arr_data[x] = dict_data[key]
        sum_data += dict_data[key]
        x += 1

    arr_data = arr_data / sum_data

    dict_model = {}
    with open("numpy_entropy_model.txt", "r") as model:
        for line in model:
            line = line.rstrip("\n")
            key, value = line.split()  # Split line into a tuple
            dict_model[key] = float(value)  # Add tuple values to dictionary

    arr_model = np.empty(len(dict_data))
    x = 0

    for key in dict_data:
        arr_model[x] = dict_model.get(key, 0)
        x += 1

    zeros = np.nonzero(arr_model == 0)

    entropy = -np.sum(arr_data * np.log(arr_data))

    if not zeros[0].size == 0:
        crossentropy = np.inf
        kl_divergence = np.inf
    else:
        crossentropy = -np.sum(arr_data * np.log(arr_model))
        kl_divergence = -np.sum(arr_data * np.log(arr_model / arr_data))

    return entropy, crossentropy, kl_divergence


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    entropy, crossentropy, kl_divergence = main(args)
    print("Entropy: {:.2f} nats".format(entropy))
    print("Crossentropy: {:.2f} nats".format(crossentropy))
    print("KL divergence: {:.2f} nats".format(kl_divergence))
