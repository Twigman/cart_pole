import numpy as np
import gym
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

# Reward threshold
max_steps_per_episode = 475
# precision
round_digits = 2
index_factor = pow(10, round_digits)


def print_q_table(q_table):
    print('\n******** Q-Table ********')
    print(q_table.shape)
    print(q_table)


def play_agent():
    total_reward = 0
    env = gym.make('CartPole-v1', render_mode='human')
    q_table = np.load('q_table_with_pole_vel_413.npy')
    print_q_table(q_table)
    
    observation = env.reset()[0]

    pole_angle = observation[2]
    pole_vel = observation[3]

    # observe the pole angle
    pole_angle_state = int(round(pole_angle, round_digits) * index_factor)
    pole_vel_state = int(round(pole_vel, round_digits) * index_factor)

    for step in range(max_steps_per_episode):
        action = np.argmax(q_table[pole_angle_state, pole_vel_state,:])
        print('action: ' + str(action))

        observation, reward, terminated, truncated, info = env.step(action)

        new_pole_angle_state = int(round(observation[2], round_digits) * index_factor)
        new_pole_vel_state = int(round(observation[3], round_digits) * index_factor)

        pole_angle_state = new_pole_angle_state
        pole_vel_state = new_pole_vel_state

        if terminated or truncated:
            break

        total_reward += reward           

    env.close()
    return total_reward

reward = play_agent()
print('total reward: ' + str(reward))
