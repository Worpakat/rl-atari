import gymnasium as gym
import ale_py
import json

env = gym.make("ALE/Riverraid-v5", render_mode="human")

obs, info = env.reset()

last_obs = None

for _ in range(2500):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    last_obs = obs

    if terminated or truncated:
        obs, info = env.reset()

env.close()


# obs_list = last_obs.tolist()    

# # Save obs as json
# with open("frame_sample.json", "w") as f:
#     json.dump(obs_list, f)

