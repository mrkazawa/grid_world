# Machine Learning HW 4 - SARSA

[![build](https://img.shields.io/badge/build-pass-green.svg)]()
[![code](https://img.shields.io/badge/code-python3.5-yellowgreen.svg)]()

This repository contains our code to answer the Machine Learning class Homework 4.
We will train agent to play **Grid World** by using the **SARSA** algorithm.
Our code is a modification based on the code available from the
[RLCode Github](https://github.com/rlcode/reinforcement-learning/tree/master/1-grid-world/4-sarsa)

## How to run

You can run the program by executing the following commands:

```shell
cd YOUR_DIR
python sarsa_agent.py i # to run scenario 1 (normal)
python sarsa_agent.py ii # to run scenario 2 (add one more triangle)
python sarsa_agent.py iii # to run scenario 3 (cliff walking env)
```

## Code Breakdown

We are going to briefly explain some of the important parts of the code that is related to the
SARSA algorithm in general.

### Agent's Global Variables

```python
self.actions = actions
self.learning_rate = 0.01
self.discount_factor = 0.9
self.epsilon = 0.1
self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
```

The `actions` are the list of available actions from the grid world, which is `['u', 'd', 'l', 'r']` up, down, left,
right respectively. The `learning rate` and `discount_factor` are the parameters to update the `q_table`. The
`epsilon` is used for the epsilon greedy approach. Lastly the `q_table` is initialized with the number of `0.0` for
all possible actions. We can think of `q_table` as `policy_table` in policy iteration algorithm.

### Epsilon Greedy -- Determine the action to the next state

```python
def get_action(self, state):
    if np.random.rand() < self.epsilon:
        # take random action
        action = np.random.choice(self.actions)
    else:
        # take action according to the q function table
        state_action = self.q_table[state]
        action = self.arg_max(state_action)
    return action
```

There are two things that may happen when an agent proceed to the `next_state` in an episode:
* **Exploration**. The epsilon greedy approach takes a `random` number and compare it with the `epsilon`. If the
`random` is smaller that `epsilon` then the agent will pick one action randomly.
* **Exploitation**. The agent will get all action from the `q_table` for
the `next_state`. Then, the agent will pick the action that has the `maximum value`.
If multiple action with the same maximum value exists, the agent will pick a random action based on
those available maximum actions.

So, the role of `epsilon` in this example is to give the agent chance to explore the environment (10% chance in this
example). If the `epsilon` is extremely low (or set to 0) than the agent will have less chance (or no chance) to
explore and the `q_table` will most likely to have `bias` from the previous episodes. Thus, we cannot see many
variance or alternatives in the episodes. In other words, the agent will not be flexible.

### Move to the next state -- Determine the reward

```python
# reward function
if next_state == self.canvas.coords(self.circle):
    reward = 100
    done = True
elif next_state in [self.canvas.coords(self.triangle1),
                    self.canvas.coords(self.triangle2)]:
    reward = -100
    done = True
else:
    reward = 0
    done = False
```

During an episode, the agent will keep on moving to the next state until it reaches either the circle grid
or the triangle grid. When the agent reaches circle, it gets `100` as a reward. However, when it reaches
triangle, it gets `-100` as a reward. Otherwise, the agent has not reached one of the possible endings. Thus,
it gets `0` as a reward.

### Updating the q_table at each step

```python
# take action and proceed one step in the environment
next_state, reward, done = env.step(action)
next_action = agent.get_action(str(next_state))

# with sample <s,a,r,s',a'>, agent learns new q function
agent.learn(str(state), action, reward, str(next_state), next_action)

state = next_state
action = next_action
```

The difference between SARSA and
[Monte Carlo](https://github.com/mrkazawa/grid_world/tree/master/monte_carlo)
method is the time when the update is conducted. SARSA updates the table after each step when the agent
move from one state to another state. Meanwhile, Monte Carlo updates only when an episode finished.

First the agent has to get the `current state (S)` and the `action (A)` from the current state.
Then, it does the action and get the `reward (R)` and the `next state (S')`.
Lastly, it needs to determine the `action (A')` for the next state.

By using all of those information, the agent can update the q_table. After the value is updated,
the `next state (S')` and the `next action (A')` will be the `current state (S)` and `current action (A)`.
Then, the program continue following the same logic.

The policy in `q_table` will be updated following the SARSA equation as follows:

```python
# with sample <s, a, r, s', a'>, learns new q function
def learn(self, state, action, reward, next_state, next_action):
    current_q = self.q_table[state][action]
    next_state_q = self.q_table[next_state][next_action]
    new_q = (current_q + self.learning_rate *
            (reward + self.discount_factor * next_state_q - current_q))
    self.q_table[state][action] = new_q
```

## Scenario 1 - Run the original code

In this first scenario, we run the original code from the
[RLCode Github](https://github.com/rlcode/reinforcement-learning/tree/master/1-grid-world/4-sarsa).

First Stage | Episode 37
:---: | :---:
![first](results/scenario1/first.png?raw=true "first") | ![last](results/scenario1/last.gif?raw=true "last")

At the left figure we can see the initialization stage of the program. The policy for each grid in the
picture will be set to `0.0` for all actions. These values will be updated after each move as the agent learning to
reach the `circle`.

The figure in the right show the states at `episode 37`.
Assuming the starting point is grid `(1,1)`, the agent moves from grid `(1,1)` to `(2,1)` to `(3,1)` to `(4,1)` to
`(5,1)` to `(5,2)` to `(4,2)` to `(4,3)` and finally `(3,3)`. This is the path that the agent have explored and learnt during
the training. We can see that this path follows the greedy policy, in which the agent will choose the action which has
higher policy.

> The grids without any numbers mean that the agent never visit those grids.

## SARSA Limitation

If you are **unlucky** the agent can be really bad at figuring out the solution for this
Grid World problem.

> **Lemma 1:** The agent can be **really bad** if he falls to the trap (triangle) at the **early stage many times**.

Some of our trials struggle with this condition. The agent keeps end up to the triangles many times
at the early stage. This makes the policy at the grid near to the triangle really bad and the agent
most likely will choose not go to that grid (**Even though it is the only way to go to the circle!**)

![loss](results/block.jpg?raw=true "loss")

Assuming that the starting point is grid `(1,1)`, then there are two crucial grids when dealing Grid World.
Those grids are grid `(3,1)` and `(1,3)`. These grids are the narrow path to reach the circle from the
starting point.

> **Lemma 2:** The path at grid (3,1) can be blocked when the agent at grid (3,1) moves to grid (3,2).
This action will make the policy to move **right** at grid (2,1) fall drastically if the policy to move
**left** at grid (4,3) contains high number.

> **Lemma 3:** The path at grid (1,3) can be blocked when the agent at grid (1,3) moves to grid (2,3).
This action will make the policy to move **down** at grid (1,2) fall drastically if the policy to move
**up** at grid (3,4) contains high number.

When the agent keeps falling to the traps, **those grids can be blocked** as depicted
in the Figure above.

* The policy to move **right** at grid `(2,1)` is `-0.02`, which is the lowest value
compared to the other alternative actions. This makes the path to grid `(3,1)` is **blocked**.
* The policy to move **down** at grid `(1,2)` is also `-0.02`, which is the lowest value
among all other alternative actions. This makes the path to grid `(1,3)` is **blocked**.

The solution to break this block is that when you are at grid `(2,1)` or `(1,2)` you must be
**VERY LUCKY** not only to hit the `epsilon` but also to get the random action correct (`right` or `down`).
Moreover, we log the time for the agent to finish an episode. Some of them are in very huge numbers because of this
problem. We put three highest number time logs (in seconds) over 50 episodes below.

```
episode : 18 --- time : 178.0
episode : 43 --- time : 273.0
episode : 47 --- time : 274.0
```

## Scenario 2 - Add one more triangle

In this scenario, we introduce one new triangle to the environment. We put the location of the new triangle very close to the
starting location at grid `(2,2)`. Our original intention in choosing this location is to make the agent difficult to reach
the `circle` by continuously falling to the trap when he explores.

First Stage | Episode 26
:---: | :---:
![first](results/scenario2/first.png?raw=true "first") | ![last](results/scenario2/last.gif?raw=true "last")

At the left figure we can see the initialization stage of the program. The policy for each grid in the
picture will be set to `0.0` for all actions. These values will be updated after each move as the agent learning to
reach the `circle`.

The figure in the right show the states at `episode 26`.
Assuming the starting point is grid `(1,1)`, the agent moves from grid `(1,1)` to `(2,1)` to `(3,1)` to `(4,1)` to `(4,2)`
to `(4,3)` and finally `(3,3)`. This is the path that the agent have explored and learnt during
the training. We can see that this path follows the greedy policy, in which the agent will choose the action which has
higher policy.

Even though the agent is able to find the shortest path to solve the problems. We have to mention that
the probability that grid `(3,1)` or `(1,3)` to be blocked is higher in this scenario because of the greater chance that the
agent will fall to the three triangles.

## Scenario 3 - Cliff Walking

In this scenario, we introduce one new environment that represents the Cliff Walking problem.
We put the rectangle at the starting point at grid `(1,1)`. Three triangles at grid `(2,1)`, `(3,1)`, `(4,1)` respectively.
Finally, we place the circle at grid `(5,1)`. The goal of this environment is the same to the Grid World.
Move the rectangle to the circle and get `100` reward. However, when the rectangle reaches the triangle, it gets
`-999` reward.

To get the best result, we need to modify the `epsilon` to be dynamic. During the early episodes, we set the `epsilon`
to a high number because we want the agent to understand the new environment really well. The `epsilon` will be decreased
as the episodes increasing. Finally, after `400 episodes`, we set the `epsilon` to be `0`. This will make the agent
become in a full exploit mode without any exploration.

```python
if episode <= 100:
    agent.epsilon = 0.5
elif 100 < episode <= 200:
    agent.epsilon = 0.4
elif 200 < episode <= 300:
    agent.epsilon = 0.2
elif 300 < episode <= 400:
    agent.epsilon = 0.1
if episode > 400:
    agent.epsilon = 0
```

First Stage | Episode 403
:---: | :---:
![first](results/scenario3/first.png?raw=true "first") | ![last](results/scenario3/last.gif?raw=true "last")

At the left figure we can see the initialization stage of the program. The policy for each grid in the
picture will be set to `0.0` for all actions. These values will be updated after each move as the agent learning to
reach the `circle`.

The figure in the right show the states at `episode 403`.
Assuming the starting point is grid `(1,1)`, the agent moves from grid `(1,1)` to `(1,2)` to `(1,3)` to `(1,4)` to `(1,5)`
to `(2,5)` to `(3,5)` to `(4,5)` to `(5,5)` to `(5,4)` to `(5,3)` to `(5,2)` and finally `(5,1)`.
This is the path that the agent have explored and learnt during the training.
We can see that this path follows the greedy policy, in which the agent will choose the action which has
higher policy.

> In this example. we can clearly see that the policy is updated when the rectangle moves through the particular state