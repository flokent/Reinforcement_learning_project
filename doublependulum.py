# Import libraries
import numpy as np
import gymnasium as gym
import cv2
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
from matplotlib import pyplot as plt
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation, FrameStack

# Import gym environment
env = gym.make('Acrobot-v1',render_mode="rgb_array")

# Initializing the state space
positions = [0, 1, 2, 3, 4, 5]

OBS_HIGH = np.array([env.observation_space.high[i] for i in positions])
OBS_HIGH[0] = 1.01
OBS_HIGH[1] = 1.01
OBS_HIGH[2] = 1.01
OBS_HIGH[3] = 1.01
OBS_HIGH[4] = 13
OBS_HIGH[5] = 29
OBS_LOW = np.array([env.observation_space.low[i] for i in positions])
OBS_LOW[0] = -1.01
OBS_LOW[1] = -1.01
OBS_LOW[2] = -1.01
OBS_LOW[3] = -1.01
OBS_LOW[4] = -13
OBS_LOW[5] = -29


# Initialization hyperparameters and Q table
N_chunks = 10
discrete_os_size = [N_chunks]*6
window_size = (OBS_HIGH - OBS_LOW)/discrete_os_size
Q_TABLE = np.random.uniform(low = -10, high = 0, size = (discrete_os_size + [env.action_space.n]))
Learning_rate = 0.1
Discount = 0.95
EPISODES = 20000
RENDER_NUMBER = EPISODES
AVERAGE_NUMBER = 100
EPSILON = 1
EPSILON_max = 1#
STOP_EXPLORING = EPISODES

# To save rewards history
ep_rewards = []
aggr_ep_rewards = {'ep':[], 'avg':[], 'min':[], 'max':[]}

# Function to transform from continuous to discrete state
def get_simplified_state(state,positions):
    modified_state = np.array([state[i] for i in positions])
    simplified_state = (modified_state - OBS_LOW)/window_size
    return tuple(simplified_state.astype(int))

# Perform simulation for EPISODES number of episodes
for episode in range(1,EPISODES+1):
    episode_reward = 0  # Initialize episode reward
    if episode % RENDER_NUMBER == 0: # Animation of the pendulum is shown every RENDER_NUMBER episodes
        render = True
    else:
        render = False
    simplified_state = get_simplified_state(env.reset()[0],positions) # Get discrete state
    dead = False # Initialize agent as not 'dead' (meaning still active)
    while not dead: # While still active
        if np.random.random() > EPSILON or episode > STOP_EXPLORING:
            # Exploitation (follow best state-action pair)
            action = np.argmax(Q_TABLE[simplified_state])
        else:
            # Exploration (try random action)
            action = np.random.randint(0, env.action_space.n)
        # Feed state and action to the dynamics. Get future state and reward
        new_state, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward # Update episode's reward
        new_simplified_state = get_simplified_state(new_state,positions) # Transform to discrete state
        if render: # Show animation if need be
            img = cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR)
            cv2.imshow("test", img)
            cv2.waitKey(50)
        dead = terminated or truncated # Is episode finished?
        if not dead: # If episode not finished
            max_future_q = np.max(Q_TABLE[new_simplified_state]) # Get Q-value of future optimal state-action pair
            current_Q = Q_TABLE[simplified_state + (action, )] # Get current state-action pair Q-value
            new_q = (1 - Learning_rate) * current_Q + Learning_rate * (reward + Discount * max_future_q) # Update current Q-value
            Q_TABLE[simplified_state+(action, )] = new_q # Save the updated current Q-value into the Q-table
        elif -new_state[0]-(new_state[0]*new_state[2]-new_state[1]*new_state[3]) > 1.0: # If episode finished and target height achieved
            Q_TABLE[simplified_state+(action, )] = 0 # Set current state-action pair to 0
        simplified_state = new_simplified_state # Update current state to the new state
    ep_rewards.append(episode_reward) # Save reward
    if episode >= 1 and episode < STOP_EXPLORING: # Decay the exploration to exploitation ratio
        EPSILON = EPSILON_max*(1-episode/EPISODES)
    else:
        EPSILON = 0
    if not episode % AVERAGE_NUMBER: # Save rewards in history
        average_reward = sum(ep_rewards[-AVERAGE_NUMBER:])/len(ep_rewards[-AVERAGE_NUMBER:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-AVERAGE_NUMBER:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-AVERAGE_NUMBER:]))
env.close()

# Save rewards and Q-table for future postprocessing
np.save(f"Stored_simulations/{EPISODES}-{N_chunks}-{np.int32(1/Learning_rate)}-{np.int32(Discount*100)}-{np.int32(EPSILON_max*100)}-qtable.npy",Q_TABLE)
np.save(f"Stored_simulations/{EPISODES}-{N_chunks}-{np.int32(1/Learning_rate)}-{np.int32(Discount*100)}-{np.int32(EPSILON_max*100)}-avg.npy",aggr_ep_rewards['avg'])
np.save(f"Stored_simulations/{EPISODES}-{N_chunks}-{np.int32(1/Learning_rate)}-{np.int32(Discount*100)}-{np.int32(EPSILON_max*100)}-min.npy",aggr_ep_rewards['min'])

# Plot reward history
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")
plt.legend(loc=4)
plt.show()
