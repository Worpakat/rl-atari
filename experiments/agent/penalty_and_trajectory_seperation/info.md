**Application Order: 3**

* **!!! BEFORE `torch.no_grad()` FIX !!!** 

* First try of removing static / uncontrollable frame sequences. 
* Firts try of application of death penalty. 
* **!** For this new modifications to be able to work, we had to change last controllable states' rewards with penalty. Those states are singular trajectories' death states. We assigned reward, but we forgot to append that last viable transition to queue. This experiment conducted while that bug was exist.
* But fixed experiment is the second one with same experiment name in experiments directory.
