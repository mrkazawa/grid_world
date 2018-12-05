# Machine Learning HW 3 - Value Iteration

[![build](https://img.shields.io/badge/build-pass-green.svg)]()
[![code](https://img.shields.io/badge/code-python3.5-yellowgreen.svg)]()

This repository contains our code to answer the Machine Learning class Homework 3. We will train agent to play **Grid World** by using the **value iteration** algorithm. Our code is a modification based on the code available from the [RLCode Github](https://github.com/rlcode/reinforcement-learning/tree/master/1-grid-world/2-value-iteration)

## How to run

You can run the program by executing the following commands:

```shell
cd YOUR_DIR
python value_iteration.py i # to run scenario 1
python value_iteration.py ii # to run scenario 2
python value_iteration.py iii # to run scenario 3
```

## Scenario 1 - Run the original code

The first scenario is to run the program from the original source code. Below is the screenshot comparison during the `initialization stage` (left) and the `final stage` when the algorithm converge (right)

First Stage | Last Stage
:---: | :---:
![1 First](capture/1_first.png?raw=true "1_first") | ![1 Final](capture/1_final.png?raw=true "1_final")

```Time to converge: 6 iterations```

At the first stage, all of the grids (or the `state`) will be initialize with the value of `0`. This value will be updated in every iteration which is one click of `Calculate` button. After the value function is converged, assuming that the starting location grid is `(1,1)` that is `(row,col)`, then we have two options:

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

## Scenario 2 - Randomize value_table initialization

In this second scenario, we want to modify the initial value of the value_table from `0` to random distribution. We use this code to generate random value_table

```python
def generate_random_value_table(self, width, height):
    value_table = [[round(random.random(), 2)] *
                    width for _ in range(height)]
    return value_table
```

Below is the screenshot comparison during the `initialization stage` (left) and the `final stage` when the algorithm converge (right)

First Stage | Last Stage
:---: | :---:
![2 First](capture/2_first.png?raw=true "2_first") | ![2 Final](capture/2_final.png?raw=true "2_final")

```Time to converge: 9 iterations (most of the time)```

As we can see from the figure, the initial value is randomize `between 0 to 1`. Due to this randomization, the convergence time takes longer. The value iteration algorithm is struggling adjusting the value for each state. Furthermore, because the initial value is random, we observe sometimes the iteration can go up to 12 iterations before it converges. **Thus, we can conclude that initial value randomization is a bad idea for value iteration algorithm.**

## Scenario 3 - Add one more obstacle

In this last scenario, we want to modify the environment by introducing one more additional obstacle into the Grid World. From the original source code, we add another obstacle in one grid below the reward grid, where the circle grid exists.

```python
self.reward[3][2] = -1  # reward -1 for triangle
```

Below is the screenshot comparison during the `initialization stage` (left) and the `final stage` when the algorithm converge (right)

First Stage | Last Stage
:---: | :---:
![3 First](capture/3_first.png?raw=true "3_first") | ![3 Final](capture/3_final.png?raw=true "3_final")

```Time to converge: 8 iterations```

This scenario takes longer time to calculates all the values from each state, especially the bottom grids due to additional obstacle there. Assuming that the starting location grid is `(1,1)` that is `(row,col)`, then the value in the grid `(3,1)` will be last one to be updated. Furthermore, the `convereged value function` is different compared to Scenario #1. The only best option now is to move to the right directly from the start following the `red line` towards the destination as shown in the `Last Stage` figure.

## Authors

**Oktian Yustus Eko** - *Initial Work* - [mrkazawa](https://github.com/mrkazawa)