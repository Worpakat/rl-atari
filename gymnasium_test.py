import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation
import ale_py
import json

env = gym.make("ALE/Riverraid-v5", render_mode="human")
env = GrayscaleObservation(env)


obs, info = env.reset()


for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    # print(obs.shape)    
    # print(obs.nbytes)
    # print(obs.dtype)    

    if terminated or truncated:
        obs, info = env.reset()

env.close()


# obs_list = last_obs.tolist()    

# # Save obs as json
# with open("frame_sample.json", "w") as f:
#     json.dump(obs_list, f)

