import numpy as np
import random
import gym
import pygame
from gym.utils.play import play

'''
Exercise for reinforcement learning

Used Algorithm: Q-Learning

Find optimal q* for every state-action pair (s, a).
Updates q* using the bellman equation.

--------------------------------------------------------

Reward of +1 for every step
Reward threshold: 475

The episode ends if any one of the following occurs:

Termination: Pole Angle is greater than ±12° (-.2095, .2095)
Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
Truncation: Episode length is greater than 500 (200 for v0)

https://www.gymlibrary.dev/environments/classic_control/cart_pole/
'''
num_episodes = 10000
# Reward threshold
max_steps_per_episode = 475

# alpha
learning_rate = 0.1
# gamma
discount_rate = 0.99

# 100% exploration
exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
# slows down exploration with every episode
exploration_decay_rate = 0.001
# precision
round_digits = 2
index_factor = pow(10, round_digits)

rewards_all_episodes = []
state_list = set()


def print_q_table(q_table):
    print('\n******** Q-Table ********')
    print(q_table.shape)
    direction_q_table = np.split(q_table, 2)
    print('right:')
    print(direction_q_table[0])
    print('-------------------------------------')
    #print(q_table)
    print('left:')
    print(direction_q_table[1])


def play_agent():
    total_reward = 0
    env = gym.make('CartPole-v1', render_mode='human')
    q_table = np.loadtxt('q_table.txt')
    print_q_table(q_table)
    
    observation = env.reset()[0]
    # observe the pole angle
    state = int(round(observation[2], round_digits) * index_factor)

    for step in range(max_steps_per_episode):
        action = np.argmax(q_table[state,:])
        print('action: ' + str(action))

        observation, reward, terminated, truncated, info = env.step(action)
        new_state = int(round(observation[2], round_digits) * index_factor)
            #print('init obs: ' + str(observation[2]) + ' -> ' + str(round(observation[2], round_digits)) + ' -> ' + str(int(round(observation[2], round_digits) * index_factor)))

            #print('new_state: ' + str(new_state))

            #print('value: ' + str(pre) + ' -> ' + str(q_table[state, action])) 
        print('state: ' + str(state) + ' -> ' + str(new_state))
        state = new_state

        if terminated or truncated:
            break

        total_reward += reward           

    env.close()
    return total_reward

reward = play_agent()
print('total reward: ' + str(reward))
