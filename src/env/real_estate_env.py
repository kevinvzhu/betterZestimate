import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class RealEstateEnv(gym.Env):
    def __init__(self, data, price_min, price_max):
        super(RealEstateEnv, self).__init__()
        
        self.data = data
        self.current_idx = 0
        self.price_min = price_min
        self.price_max = price_max

        self.features = list(self.data.columns)
        self.features.remove('price')

        # Observation space: normalized features (already scaled to [0,1])
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(self.features),), dtype=np.float32)
        
        # Action space: predict price in normalized scale [0,1]
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_idx = 0
        return self._get_state(), {}

    def step(self, action):
        # Here action is expected to be in [0,1], thanks to RescaleAction wrapper
        predicted_price = action[0] * (self.price_max - self.price_min) + self.price_min
        true_price = self.data.iloc[self.current_idx]['price']

        error = abs(predicted_price - true_price)
        # Reward: a simple negative MSE-based approach
        reward = -error

        self.current_idx += 1
        done = self.current_idx >= len(self.data)
        next_state = self._get_state() if not done else None

        info = {"true_price": true_price, "predicted_price": predicted_price, "error": error}
        return next_state, reward, done, False, info

    def _get_state(self):
        state = self.data.iloc[self.current_idx][self.features].values.astype(np.float32)
        return state