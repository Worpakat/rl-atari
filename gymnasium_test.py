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

obs_list = []


for i in range(70):
    if i == 0: print("START!")

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if current_lives != info["lives"]:
        current_lives = info["lives"]
        print(f"Current lives: {current_lives}")

    # if current_lives == 1 or current_lives == 0:
    #     if terminated or truncated:
    #         print("Episode finished")
    #     obs_list.append(obs)

    if terminated or truncated:
        obs, info = env.reset()

env.close()

# obs_list = obs_list[-30:]
# obs_list = [obs.tolist() for obs in obs_list]

# # Save obs as json
# with open("episode_end_frames.json", "w") as f:
#     json.dump(obs_list, f)

