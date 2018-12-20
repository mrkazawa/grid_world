# Reinforcement Learning - Grid World

[![build](https://img.shields.io/badge/build-pass-green.svg)]()
[![code](https://img.shields.io/badge/code-python3.5-yellowgreen.svg)]()

This repository contains our collection codes to solve the Grid World problem.

## Dependency

This repository depends on the following libraries:

```shell
pip install numpy
pip install matplotlib
```

## What is Grid World

![env](gridworld.png?raw=true "env")

Grid World is an environment where there are one `rectangle`, one `circle`, and two `triangles`.
We can only move the rectangle. The movement can only be: `up`. `down`, `left` and `right`.
The goal of this environment is the user has to move the `rectangle` to the `circle`.
However, the rectangle needs to avoid the `triangle` when closely move to the `circle`.
In this repository, we are teaching the `rectangle` to reach the `circle` by using **classical reinforcement learning**.

## List of projects

The list of reinforcement learning projects available in this repository is as follows:

1. [Value Iteration](https://github.com/mrkazawa/grid_world/tree/master/value_iteration)
2. [Policy Iteration](https://github.com/mrkazawa/grid_world/tree/master/policy_iteration)
3. [Monte Carlo](https://github.com/mrkazawa/grid_world/tree/master/monte_carlo)
4. [SARSA](https://github.com/mrkazawa/grid_world/tree/master/sarsa)
5. [Q Learning](https://github.com/mrkazawa/grid_world/tree/master/q_learning)

## Feedback

If you have any issues, requests, feedbacks just **create an Issue** on Github.
*Want to contribute?* Just create a **Pull request**.