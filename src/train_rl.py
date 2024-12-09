from stable_baselines3 import PPO
from env_setup import RealEstatePortfolioEnv

if __name__ == "__main__":
    property_data_path = "/Users/KZJer/Documents/GitHub/betterZestimate/data/processed_data.csv"
    market_data_path = "/Users/KZJer/Documents/GitHub/betterZestimate/data/market_data.csv"

    env = RealEstatePortfolioEnv(
        property_data_path=property_data_path,
        market_data_path=market_data_path,
        initial_cash=100_000_000,
        num_properties=5
    )

    model = PPO("MlpPolicy", env,
                verbose=1,
                n_steps=4096,
                batch_size=256,
                gamma=0.95,
                learning_rate=1e-4)
    model.learn(total_timesteps=300000)  # More training steps since we have more data
    model.save("re_portfolio_ppo_new")

    # Evaluate final performance
    obs, info = env.reset()
    done = False
    step_idx = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        print(f"Step {step_idx}, Action chosen: {action}")
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {step_idx}, Portfolio value: {info['portfolio_value']}, Reward: {reward}, Owned: {env.owned_properties}")
        step_idx += 1

    print("Final portfolio value:", info["portfolio_value"])


    # Do-nothing means always choose "hold" (action=0 for each property)
    # obs, info = env.reset()
    # done = False
    # total_reward = 0
    # while not done:
    #     # Action: zero for every property (hold)
    #     action = [0]*env.num_properties
    #     obs, reward, done, truncated, info = env.step(action)
    #     total_reward += reward

    # print("Do-Nothing final portfolio value:", info["portfolio_value"])
    # print("Do-Nothing total cumulative reward:", total_reward)

    # Random Policy
    # import random

    # obs, info = env.reset()
    # done = False
    # total_reward = 0
    # while not done:
    #     # Randomly choose 0,1,2 for each property
    #     action = [random.randint(0,2) for _ in range(env.num_properties)]
    #     obs, reward, done, truncated, info = env.step(action)
    #     total_reward += reward

    # print("Random final portfolio value:", info["portfolio_value"])
    # print("Random total cumulative reward:", total_reward)
