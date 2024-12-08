import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class RealEstateEnv(gym.Env):
    def __init__(self, data, price_min, price_max):
        super(RealEstateEnv, self).__init__()

        # Use the preprocessed data directly
        self.data = data
        self.current_idx = 0
        self.price_min = price_min
        self.price_max = price_max

        # State space: Use all columns except 'price' (target variable)
        self.features = list(self.data.columns)
        self.features.remove('price')

        # Observation space: Shape based on number of features
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(len(self.features),), dtype=np.float32
        )

        # Action space: Continuous price prediction
        self.action_space = spaces.Box(low=price_min, high=price_max, shape=(1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        # Optionally set the seed for reproducibility
        if seed is not None:
            np.random.seed(seed)

        self.current_idx = 0
        state = self._get_state()
        info = {}  # Additional info dictionary required by Gymnasium
        return state, info

    def step(self, action):
        """
        Executes a single step in the environment.

        Parameters:
            action (array-like): The normalized price prediction in the range [0, 1].

        Returns:
            next_state (array-like): The next state of the environment.
            reward (float): The calculated reward for the current action.
            done (bool): Whether the episode has ended.
            truncated (bool): Whether the episode was truncated (always False here).
            info (dict): Additional debugging information.
        """

        # Map normalized action to real-world price
        predicted_price = action[0] * (self.price_max - self.price_min) + self.price_min

        # Get the true price for the current property
        true_price = self.data.iloc[self.current_idx]['price']

        # Calculate reward
        error = abs(predicted_price - true_price)
        threshold = 0.1 * (self.price_max - self.price_min)  # 10% of the price range
        if error <= threshold:
            reward = 1 - (error / threshold)  # Higher reward for smaller errors
        else:
            reward = -1  # Large penalty for large errors

        # Move to the next property
        self.current_idx += 1
        done = self.current_idx >= len(self.data)

        # Get the next state or terminate
        next_state = self._get_state() if not done else None

        # Debugging: Check action bounds
        if not (0 <= action[0] <= 1):
            print(f"Action out of bounds: {action[0]}")

        # Additional information (can include debugging metrics or diagnostics)
        info = {"true_price": true_price, "predicted_price": predicted_price, "error": error}

        # Return results
        return next_state, reward, done, False, info
    def _get_state(self):
        # Get the current row of feature values (exclude 'price')
        state = self.data.iloc[self.current_idx][self.features].values.astype(np.float32)
        if np.isnan(state).any():
            print(f"NaN detected in state at index {self.current_idx}")
        if np.isinf(state).any():
            print(f"Infinite value detected in state at index {self.current_idx}")
        return state