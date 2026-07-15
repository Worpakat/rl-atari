import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, RecordEpisodeStatistics
import ale_py
import json

env = gym.make("ALE/Riverraid-v5", render_mode="human")
env = GrayscaleObservation(env)
env = RecordEpisodeStatistics(env)

obs, info = env.reset()
current_lives = env.unwrapped.ale.lives()
print(f"Current lives: {current_lives}")

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if current_lives != info["lives"]:
        current_lives = info["lives"]
        print(f"Current lives: {current_lives}")

    if terminated or truncated:
        obs, info = env.reset()

env.close()


# obs_list = last_obs.tolist()    

# # Save obs as json
# with open("frame_sample.json", "w") as f:
#     json.dump(obs_list, f)

