# Machine Learning HW 3 - Policy Iteration

[![build](https://img.shields.io/badge/build-pass-green.svg)]()
[![code](https://img.shields.io/badge/code-python3.5-yellowgreen.svg)]()

This repository contains our code to answer the Machine Learning class Homework 3. We will train agent to play **Grid World** by using the **policy iteration** algorithm. Our code is a modification based on the code available from the [RLCode Github](https://github.com/rlcode/reinforcement-learning/tree/master/1-grid-world/1-policy-iteration)

## How to run

You can run the program by executing the following commands:

```shell
cd YOUR_DIR
python policy_iteration.py i # to run scenario 1
python policy_iteration.py ii # to run scenario 2
python policy_iteration.py iii # to run scenario 3
```

## Scenario 1 - Run the original code

The first scenario is to run the program from the original source code. Below is the screenshot comparison during the `initialization stage` (left) and the `final stage` when the algorithm converge (right)

First Stage | Last Stage
:---: | :---:
![1 First](capture/1_first.png?raw=true "1_first") | ![1 Final](capture/1_final.png?raw=true "1_final")

* At the first stage, all of the grids (or the `state`) will be initialize with the value of `0`. This value will be updated in every iteration which is one click of `Evaluate` button.
* The policy will initialized with the value of `0.25` for all actions. Every action has the same chance to be utilized to update the value. The policy will be updated in every click of `Improve` button.

To get the most efficient result, we need to take turn in using `Evaluate` and `Improve` button. A user can click this button arbitrarily. Therefore, there is many combinations to iterate value and policy to achieve convergence. We run simple test to evaluate those combinations.

Evaluate | Improve | Iteration to Converge
:---: | :---: | :---:
1 click | 1 click | 6
2 click | 1 click | 4
3 click | 1 click | 3
4 click | 1 click | 3
5 click | 1 click | 3

We can see that the more frequent we update the value, the lesser the iteration needed to converge. We define iteration as `one click of Improve button, which is similar to one time policy update`. However, at some point updating the value more and more will not be efficient. As we see from table above that updating value 3 times in one iteration generates similar result as 5 times.

After the value function is converged, assuming that the starting location grid is `(1,1)` that is `(row,col)`, then we have two options:

* move to grid `(1,2)` then following the red line to the destination.
* move to grid `(2,1)` then following the blue line to the destination.

These two options are possible becuase the world is symetricall from the distance perspective between the starting location to the destination. Thus, it will result in two shortest paths, 6 hops from start to finish. Both of the paths will have the same set of grids with the same value as follows:

```
             1.0  (6th grid)
1.0  * 0.9 = 0.9  (5th grid)
0.9  * 0.9 = 0.81 (4th grid)
0.81 * 0.9 = 0.73 (3rd grid)
0.73 * 0.9 = 0.66 (2nd grid)
0.66 * 0.9 = 0.59 (1st grid)
```

The grids besides the destination, which are `(3,4)` and `(4,3)` will have the value of `1` because it is very close to the reward. In the code, we use the discount factor of `0.9` that is why the value in the next hop will be discounted by `0.9` as shown in the calculation above.

## Scenario 2 - Randomize value_table and policy_table initialization

In this second scenario, we want to modify the initial value of the value_table from `0` to random distribution between `0 to 1`. We use this code to generate random value_table

```python
def generate_random_value_table(self, width, height):
    value_table = [[round(random.random(), 2)] *
                    width for _ in range(height)]
    return value_table
```

Furthermore, we want to modify the initial value of the policy_table from `0.25` to random distribution that has the sum of `1` (because policy is represented as a probability). We use this code to generate random policy_table

```python
def generate_random_policy_table(self, width, height):
    r = list(np.random.dirichlet(np.ones(4), size=1).ravel())
    policy_table = [[r] * width for _ in range(height)]
    return policy_table
```

Below is the screenshot comparison during the `initialization stage` (left) and the `final stage` when the algorithm converge (right)

First Stage | Last Stage
:---: | :---:
![2 First](capture/2_first.png?raw=true "2_first") | ![2 Final](capture/2_final.png?raw=true "2_final")

Evaluate | Improve | Iteration to Converge
:---: | :---: | :---:
1 click | 1 click | 6
2 click | 1 click | 4
3 click | 1 click | 3
4 click | 1 click | 3
5 click | 1 click | 3

As we can see from the figure, the initial value is randomize `between 0 to 1`. Then we test the program with different value and policy combinations. The results are shown in the figure. Surprisingly, those results are not different with the one from Scenario #1.

## Scenario 3 - Add one more obstacle

In this last scenario, we want to modify the environment by introducing one more additional obstacle into the Grid World. From the original source code, we add another obstacle in one grid below the reward grid, where the circle grid exists.

```python
self.reward[3][2] = -1  # reward -1 for triangle
```

Below is the screenshot comparison during the `initialization stage` (left) and the `final stage` when the algorithm converge (right)

First Stage | Last Stage
:---: | :---:
![3 First](capture/3_first.png?raw=true "3_first") | ![3 Final](capture/3_final.png?raw=true "3_final")

Evaluate | Improve | Iteration to Converge
:---: | :---: | :---:
1 click | 1 click | 8
2 click | 1 click | 5
3 click | 1 click | 4
4 click | 1 click | 4
5 click | 1 click | 4

Assuming that the starting location grid is `(1,1)` that is `(row,col)`, then the value in the grid `(3,1)` will be last one to be updated. Furthermore, the `convereged value function` is different compared to Scenario #1. The only best option now is to move to the right directly from the start following the `red line` towards the destination as shown in the `Last Stage` figure.

## Authors

**Oktian Yustus Eko** - *Initial Work* - [mrkazawa](https://github.com/mrkazawa)