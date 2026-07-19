# First Experiment After `torch.no_grad()` Fix.

**Problem:** We were gather gradients for either encoder and keys out of the `_network_optimization_step()` via `NECAgent.encode()` and variety of operations with `DND.keys`. This was an unintended way for training to process. Also causes serious unstability.

Eventually handled those parts with ``torch.no_grad()`` context managers. Eventually, we've got really more stabil and obviously working well training process and results than previous experiments. Although, it did not solve whole stability problems and training problems. But, definetely it has handled an logical bug and got training into intended way of processing. 

Next natural step is tuning and experimenting hyperparameters before we try optional update strategies.