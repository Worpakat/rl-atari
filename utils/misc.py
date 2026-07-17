from pathlib import Path
import json

import numpy as np
import torch
import scipy.signal as signal

from models.transition_classes import Transition


def ensure_directory(path: str | Path) -> Path:
    """
    Creates a directory if it does not already exist.

    Parameters
    ----------
    path:
        Directory path.

    Returns
    -------
    Path
        Path object pointing to the directory.
    """

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    return path


def discount(x, gamma):
  """
  Compute discounted sum of future values
  out[i] = in[i] + gamma * in[i+1] + gamma^2 * in[i+2] + ...
  """
  return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1].copy()


##----------------TEST_FUNCTIONS------------------------
def print_and_save_death_transitions(transitions: list[Transition]):
    
    for i, transition in enumerate(transitions):
        print(transition.reward)
        
        with open(f"death_transition_{i}.json", "w") as f:
            json.dump(transition.state.tolist(), f)


    

