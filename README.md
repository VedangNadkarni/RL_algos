# RL_algos

This repository contains experimental code. The code trains an Actor-Critic Model
using only visual data (not the states). Hence this method can be more easily extended
to more complex environments. The environment contains all the necessary dependecies,
but there may be a few unnecessary ones too (but likely not very large ones).

The Actor-Critic model is trained using a GAE policy gradient method, which is an
improvement on the basic Actor-Critic policy gradient. The OpenAI spinningup demo
code was used as a base and modifications were made on top of it. The tensorboards
files contain the training progress and data. Code from other sources may have also
been used in addition to the base. However, it required considerable effort to adapt
the base code to the CNN method (visual training).

To check it out, clone the repo, install dependencies using conda. There may be mujoco
errors, since there is some work being done on MuJoCo, however not complete. You can
find the mujoco installation procedure at https://github.com/openai/mujoco-py.

The vpg function has tensorboard functionality, so launch tensorboard to check that out.
