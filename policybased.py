import numpy as np
import pickle as pickle
import tensorflow as tf

import matplotlib.pyplot as plt
import math

import gym
env = gym.make("CartPole-v0")

if False:
    env.reset()
    random_episodes = 0
    reward_sum = 0
    while random_episodes < 10:
        env.render()
        observation, reward, done, _ = env.step(env.action_space.sample())
        reward_sum += reward
        if done:
            random_episodes += 1
            print("Reward for this episode was: " + str(reward_sum))
            reward_sum = 0
            env.reset()

H = 10 # number of hidden layer neurons
batch_size = 5 # every how many episodes do param update?
learning_rate = 1e-2 # feel free to play with this to train faster/stably
gamma = 0.99 # discount factor for reward
D = 4 # input dimensionality

tf.reset_default_graph()

# Define network as it goes from observation of environment to probability
# of choosing the action left or right
observations = tf.placeholder(tf.float32, [None,D], name="input_x")
W1 = tf.get_variable("W1", shape=[D, H],
                     initializer=tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(observations, W1))
W2 = tf.get_variable("W2", shape=[H, 1],
                     initializer=tf.contrib.layers.xavier_initializer())
score = tf.matmul(layer1, W2)
probability = tf.nn.sigmoid(score)

# Define parts of the network needed for learning good policy
tvars = tf.trainable_variables()
input_y = tf.placeholder(tf.float32, [None,1], name="input_y")
advantages = tf.placeholder(tf.float32, name="reward_signal")

# Loss function, sends weights in the direction of making actions
# that give good advantage (reward over time) more likely, and actions that didn't
# less likely.
loglik = tf.log(input_y*(input_y - probability) + (1 - input_y) * (input_y + probability))
loss = -tf.reduce_mean(loglik * advantages)
newGrads = tf.gradients(loss, tvars)

# Once we have collected a series of gradients from multiple episodes, we apply them.
# We don't just apply gradients after every episode in order to account for
# noise in the reward signal.
adam = tf.train.AdamOptimizer(learning_rate=learning_rate) # Our optimizer
# Placeholders to send the final gradient through when we update.
W1Grad = tf.placeholder(tf.float32, name="batch_grad1") 

W2Grad = tf.placeholder(tf.float32, name="batch_grad2")
batchGrad = [W1Grad, W2Grad]
updateGrads = adam.apply_gradients(zip(batchGrad, tvars))

def discount_rewards(r):
    """Take 1D float array of rewards and compute discounted reward"""
    discounted_r = np.zeros_like(r)
    running_add = 0
    
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

xs, hs, dlogps, drs, ys, tfps = [], [], [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 1
total_episodes = 10000
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    rendering = False
    sess.run(init)
    observation = env.reset() # Obtain an initial observation of the environment
    # Reset the gradient placeholder. We will collect gradients in gradBuffer
    # until we are ready to update our policy network.
    gradBuffer = sess.run(tvars)
    for ix,grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

    while episode_number <= total_episodes:

        good_agent = running_reward is not None and running_reward/batch_size > 190
        # it once our agent is doing a good job.
        if good_agent or rendering:
            env.render()
            rendering = True

        # Make sure the observation is in the shape the network can handle
        x = np.reshape(observation, [1, D])

        # Run the policy network and get an action to take.
        tfprob = sess.run(probability, feed_dict={observations: x})
        action = 1 if np.random.uniform() < tfprob else 0
        xs.append(x)
        y = 1 if action == 0 else 0
        ys.append(y)

        # step the environment and get new measurements
        observation, reward, done, info = env.step(action)
        reward_sum += reward
        # record reward(has to be done after we call step() to get reward
        # for previous actions)
        drs.append(reward)

        if not good_agent and done:
            episode_number += 1
            # stack together all inputs, hidden states, action gradients, and rewards
            # for this episode
            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(drs)
            tfp = tfps
            xs, hs, dlogps, drs, ys, tfps = [], [], [], [], [], []

            # compute the discounted reward backwards through time
            discounted_epr = discount_rewards(epr)
            # size the rewards to be unit normal (helps control the gradient
            # estimator variance)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)

            # Get the gradient for this episode, and save it in the gradbuffer
            tGrad = sess.run(newGrads, feed_dict={observations: epx, input_y: epy, advantages: discounted_epr})
            for ix, grad in enumerate(tGrad):
                gradBuffer[ix] += grad

            # If we have completed enough episodes, then update the policy network with our gradients.
            if episode_number % batch_size == 0:
                sess.run(updateGrads, feed_dict={W1Grad: gradBuffer[0], W2Grad:gradBuffer[1]})
                for ix, grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0

                # Give a summary of how well our network is doing for each batch of episodes
                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                print("Average reward for episode: " + str(reward_sum / batch_size) + " total average reward: " + str(running_reward/batch_size))

                if reward_sum/batch_size > 200:
                    print("Task solved in " + str(episode_number) + " episodes!")
                    break

                reward_sum = 0

            observation = env.reset()

print(str(episode_number) + " episodes completed.")
