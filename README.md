# Code for the paper "Sub-optimal Experts mitigate Ambiguity in Inverse Reinforcement Learning" (NeurIPS 2024)
This repository contains the code to reproduce the experiments in the paper "Sub-optimal Experts mitigate Ambiguity in Inverse Reinforcement Learning" (Poiani et al., NeurIPS 2024).

We relied on `python 3.8` for the implementation. The file `requirements.txt` contains the necessary dependency.

To reproduce the experiment with perfect knowledge on the MDP, run the command `python correct_exp.py`.
To reproduce the experiment without knowledge of the mdp, run the command `python emp_exp_ub.py`. 
Within this last file it is also possible to modify the value of epsilon, delta, and the number of runs.