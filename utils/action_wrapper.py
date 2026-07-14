import gymnasium as gym
from gymnasium import spaces

## EXAMPLE USAGE
# env = gym.make("ALE/Breakout-v5")

# env = RestrictedActionWrapper(
#     env,
#     action_mapping=[0, 1, 2, 3, 4, 5],
# )

class RestrictedActionWrapper(gym.ActionWrapper):
    """
    Restricts the available discrete actions.
    """

    def __init__(
        self,
        env: gym.Env,
        action_mapping: list[int],
    ):
        super().__init__(env)

        self.action_mapping = action_mapping

        self.action_space = spaces.Discrete(len(action_mapping))

    def action(self, action: int) -> int:
        """
        Maps the agent's action to the original environment action.
        """
        return self.action_mapping[action]
    

