import pandas as pd
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from src.env.real_estate_env import RealEstateEnv
from src.preprocessing.pre_processing import preprocess_data
import os
import numpy as np

# Load and preprocess data
file_path = "data/cs238_modified_data.json"
data, scaler, encoder, price_min, price_max = preprocess_data(file_path)

# Initialize the environment
env = RealEstateEnv(data, price_min, price_max)

# Add action noise for exploration
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(1), sigma=0.1)

# Train the DDPG agent directly on the single environment
model = DDPG("MlpPolicy", env, verbose=1, action_noise=action_noise,
    learning_rate=1e-3,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=64,
    gamma=0.99,
    tau=0.005)

# Define the path for the log file
log_file_path = '/Users/KZJer/Documents/GitHub/betterZestimate/src/log_file.txt'

# Open the file in write mode
with open(log_file_path, 'w') as log_file:
    obs, info = env.reset()  # Correctly get the initial observation
    for step in range(1000):
        # Predict the action
        action, _states = model.predict(obs, deterministic=True)
        
        # Take the action in the environment
        obs, reward, done, truncated, info = env.step(action)
        
        # Log the action and reward
        log_file.write(f"Step {step+1}: Action: {action}, Reward: {reward}\n")
        
        # Reset the environment if done
        if done:
            obs, info = env.reset()

# Train the model
model.learn(total_timesteps=10000)

# Save the model
model.save("real_estate_ddpg")

# Test the model
obs, info = env.reset()  # Unpack both observation and info
print("Initial observation:", obs)
print("Initial observation shape:", obs.shape)
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    print(f"Predicted action (price): {action}")  # Print the predicted action
    obs, reward, done, truncated, info = env.step(action)
    print(f"Action: {action}, Reward: {reward}")