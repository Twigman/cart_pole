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
num_episodes = 20000
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
exploration_decay_rate = 0.0005
# precision
round_digits = 2
index_factor = pow(10, round_digits)

rewards_all_episodes = []
state_list = set()


def play_manually():
    mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 1}
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    play(env, keys_to_action=mapping)


def init_q_table():
    action_space_size = 2
    # (-.2095, .2095)
    state_space = 0.3 * 2
    # (-2, 2)
    vel_space = 2 * 2 
    state_space_size = int(state_space * index_factor)
    vel_space_size = int(vel_space * index_factor)
    q_table = np.zeros((state_space_size, vel_space_size, action_space_size))

    return q_table


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


def print_avg_reward_per(num_of_episodes=1000): 
    rewards_per_episodes = np.split(np.array(rewards_all_episodes), num_episodes/num_of_episodes)
    counter = num_of_episodes
    print('******** Average reward per ' + str(num_of_episodes) + ' episodes ********\n')
    
    for r in rewards_per_episodes:
        print(str(counter) + ': ' + str(sum(r / num_of_episodes)))
        counter += num_of_episodes


def simulation(exploration_rate):
    total_reward = 0
    env = gym.make('CartPole-v1')
    q_table = init_q_table()

    for episode in range(num_episodes):
        print('* Episode ' + str(episode + 1) + ' started *')
        #observation = env.reset(seed=seed)[0]
        observation = env.reset()[0]
        #observation = env.reset()
        pole_angle = observation[2]
        pole_vel = observation[3]

        # observe the pole angle
        pole_angle_state = int(round(pole_angle, round_digits) * index_factor)
        pole_vel_state = int(round(pole_vel, round_digits) * index_factor)
        #print('init obs: ' + str(observation[2]) + ' -> ' + str(round(observation[2], round_digits)) + ' -> ' + str(int(round(observation[2], round_digits) * index_factor)))
        #print('init state: ' + str(state))
        total_reward = 0

        for step in range(max_steps_per_episode):
            # random number between 0 and 1
            # exploreation-exploitation trade-off
            exploration_rate_threshold = random.uniform(0, 1)

            if exploration_rate_threshold > exploration_rate:
                # exploit-mode
                # choose action with highest q value for the current state
                action = np.argmax(q_table[pole_angle_state, pole_vel_state,:])
            else:
                # explore-mode
                # choose random action
                # 0 = left
                # 1 = right
                action = env.action_space.sample()

            #print('action: ' + str(action))
            # execute action
            observation, reward, terminated, truncated, info = env.step(action)
            new_pole_angle_state = int(round(observation[2], round_digits) * index_factor)
            new_pole_vel_state = int(round(observation[3], round_digits) * index_factor)
            #print('init obs: ' + str(observation[2]) + ' -> ' + str(round(observation[2], round_digits)) + ' -> ' + str(int(round(observation[2], round_digits) * index_factor)))

            #print('new_state: ' + str(new_state))

            # reduce reward
            reduced_reward = pow(discount_rate, step) * reward

            if terminated or truncated:
                reduced_reward = -1

            # update q table
            #pre = q_table[state, action]

            q_table[pole_angle_state, pole_vel_state, action] = (1 - learning_rate) * q_table[pole_angle_state, pole_vel_state, action] + learning_rate * \
                (reduced_reward + discount_rate * np.max(q_table[new_pole_angle_state, new_pole_vel_state, :]))

            #print('value: ' + str(pre) + ' -> ' + str(q_table[state, action])) 
            #print('state: ' + str(state) + ' -> ' + str(new_state))
            pole_angle_state = new_pole_angle_state
            pole_vel_state = new_pole_vel_state
            #state_list.add(state)

            if terminated or truncated:
                break

            total_reward += reward
            
        # update exploration rate
        # the probability that the agent will explore the environment, will become more unlikely with every episode
        exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * \
            np.exp(-exploration_decay_rate * episode)
        #print('new exp rate: ' + str(exploration_rate))

        rewards_all_episodes.append(total_reward)

    env.close()
    return q_table

q_table = simulation(exploration_rate)
print_avg_reward_per(num_of_episodes=1000)
print_q_table(q_table)
print('states: ' + str(state_list))

np.save('q_table_with_pole_vel.npy', q_table)



# play_manually()