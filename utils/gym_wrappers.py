import math

import gymnasium as gym
from gymnasium import spaces

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
    
## EXAMPLE USAGE
# env = gym.make("ALE/Breakout-v5")

# env = RestrictedActionWrapper(
#     env,
#     action_mapping=[0, 1, 2, 3, 4, 5],
# )

###==============================================================================

class RewardWrapper(gym.RewardWrapper):

    def __init__(
        self,
        env: gym.Env,
        strategy: str = "identity",
        parameters: dict = {},
    ):
        super().__init__(env)

        strategies = {
            "identity": self._identity,
            "scale": self._scale,
            "clip": self._clip,
            "tanh": self._tanh,
            "log": self._log,
        }

        if strategy not in strategies:
            raise ValueError(f"Unknown reward strategy: {strategy}")

        self._params = parameters
        self._reward_fn = strategies[strategy]

    def reward(self, reward):
        transport = self._params.get("transport", 0.0)
        reward += transport
        
        return self._reward_fn(reward)

    def _identity(self, reward):
        return reward

    def _scale(self, reward):
        scale = self._params.get("scale", 1.0)
        return reward * scale

    def _clip(self, reward):
        minimum = self._params.get("minimum", -1.0)
        maximum = self._params.get("maximum", 1.0)
        return max(min(reward, maximum), minimum)

    def _tanh(self, reward):
        scale = self._params.get("scale", 1.0)
        return math.tanh(reward * scale)

    def _log(self, reward):
        scale = self._params.get("scale", 1.0)
        return math.copysign(
            math.log1p(abs(reward) * scale),
            reward,
        )
    
## EXAMPLE USAGE
#----Train_Config----
# {
#     "reward_strategy": "scale",
#     "reward_parameters": {
#         "scale": 0.01
#     }
# }

#----In_code----
# environment = RewardWrapper(
#     environment,
#     strategy="scale",
#     parameters=config.reward_parameters,
# )