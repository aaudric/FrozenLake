#%%

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import imageio
import os

# Initialize the FrozenLake environment
env = gym.make('FrozenLake-v1', is_slippery=True,render_mode='rgb_array')

# Define Q-learning parameters
alpha = 0.0025  # Learning rate for the neural network
gamma = 0.95    # Discount factor
epsilon = 1.0   # Initial exploration-exploitation parameter
epsilon_min = 0.1
epsilon_decay = 0.995
episodes = 2000
max_steps = 100
batch_size = 32

# Directory to save GIFs
gif_dir = 'gifs'
os.makedirs(gif_dir, exist_ok=True)

# Define the neural network model
def create_model(input_shape, output_shape):
    model = tf.keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=input_shape),
        layers.Dense(64, activation='relu'),
        layers.Dense(output_shape, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha),
                  loss='mse')
    return model

# Initialize the model
input_shape = (env.observation_space.n,)
output_shape = env.action_space.n
model = create_model(input_shape, output_shape)
target_model = create_model(input_shape, output_shape)
target_model.set_weights(model.get_weights())

# Epsilon-greedy function to choose an action
def choose_action(state, epsilon):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    q_values = model.predict(state)
    return np.argmax(q_values[0])

# Replay buffer to store experiences
class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.buffer = []
        self.max_size = max_size

    def add(self, experience):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

replay_buffer = ReplayBuffer()

# Lists to track losses and rewards
losses = []
rewards = []

# Train the agent with Deep Q-learning
for episode in range(episodes):
    state, _ = env.reset()
    state = np.identity(env.observation_space.n)[state].reshape(1, -1)
    total_reward = 0
    frames = []
    loss = 0

    for step in range(max_steps):
        action = choose_action(state, epsilon)
        next_state, reward, done, _, _ = env.step(action)
        next_state = np.identity(env.observation_space.n)[next_state].reshape(1, -1)
        
        replay_buffer.add((state, action, reward, next_state, done))

        if len(replay_buffer.buffer) >= batch_size:
            states, actions, rewards_batch, next_states, dones = replay_buffer.sample(batch_size)

            # Reshape tensors
            states = np.vstack(states)
            next_states = np.vstack(next_states)

            q_values = model.predict(states)
            next_q_values = target_model.predict(next_states)

            targets = q_values.copy()
            for i in range(batch_size):
                if dones[i]:
                    targets[i, actions[i]] = rewards_batch[i]
                else:
                    targets[i, actions[i]] = rewards_batch[i] + gamma * np.max(next_q_values[i])

            loss = model.train_on_batch(states, targets)
            losses.append(loss)

        state = next_state
        total_reward += reward

        # Record frames for the GIF and ensure valid frames
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        if done:
            break

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    rewards.append(total_reward)

    # Save the GIF of the episode if there are frames
    if frames:
        imageio.mimsave(os.path.join(gif_dir, f'episode_{episode+1}.gif'), frames, duration=0.1)

    # Periodically update the target model
    if episode % 10 == 0:
        target_model.set_weights(model.get_weights())

    print(f"Episode: {episode+1}/{episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}, Loss: {loss:.4f}")

# Save the model
model.save('model/dqn_frozenlake_model.keras')



# %%
