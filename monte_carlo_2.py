#!/usr/bin/env python3
# d4215ecb-c593-11e8-a4be-00505601122b
# c751264b-78ee-11eb-a1a9-005056ad4f31
# bee39584-17d2-11e8-9de3-00505601122b
import argparse

import gym
import numpy as np
from numpy import random

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--episodes", default=2000, type=int, help="Training episodes.")
parser.add_argument("--epsilon", default=0.2, type=float, help="Exploration factor.")

def main(env, args):
    # Fix random seed
    np.random.seed(args.seed)
    print(type(args.episodes))
    # TODO:
    # - Create Q, a zero-filled NumPy array with shape [number of states, number of actions],
    #   representing estimated Q value of a given (state, action) pair.
    # - Create C, a zero-filled NumPy array with the same shape,
    #   representing number of observed returns of a given (state, action) pair.
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    C = np.zeros([env.observation_space.n, env.action_space.n])

    for _ in range(args.episodes):
        # Perform episode, collecting states, actions and rewards
        states, actions, rewards = [], [], []
        state, done = env.reset(), False
        while not done:
            if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                env.render()

            # TODO: Compute `action` using epsilon-greedy policy. Therefore,
            # with probability of args.epsilon, use a random action,
            # otherwise choose and action with maximum Q[state, action].
            if np.random.rand() >= args.epsilon:
                action = np.argmax(Q[state])
            else:
                action = np.random.randint(2)

            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state

        # TODO: Compute returns from the received rewards
        # and update Q and C.
        returns = 0
        for i in reversed(range(len(states))):
            returns += rewards[i]
            C[states[i], actions[i]] += 1
            Q[states[i], actions[i]] += (returns - Q[states[i], actions[i]]) / C[states[i], actions[i]]

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True), False
        while not done:
            # TODO: Choose greedy action
            action = np.argmax(Q[state])
            state, reward, done, _ = env.step(action)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationWrapper(wrappers.DiscreteCartPoleWrapper(gym.make("CartPole-v1")), args.seed)

    main(env, args)
