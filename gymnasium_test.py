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

counter = 0
action = 1

for i in range(1000):
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(2)

    if current_lives != info["lives"]:
        current_lives = info["lives"]
        print(f"Current lives: {current_lives}")


    obs_list.append(obs)

    if terminated or truncated:
        print("Episode ended")

        # obs_list = [obs.tolist() for obs in obs_list]

        # # Save obs as json
        # with open("episode_frames.json", "w") as f:
        #     json.dump(obs_list, f)



        obs, info = env.reset()

env.close()

# obs_list = obs_list[-30:]
# obs_list = [obs.tolist() for obs in obs_list]

# # Save obs as json
# with open("episode_end_frames.json", "w") as f:
#     json.dump(obs_list, f)

