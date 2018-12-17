# Machine Learning HW 4 - Monte Carlo

[![build](https://img.shields.io/badge/build-pass-green.svg)]()
[![code](https://img.shields.io/badge/code-python3.5-yellowgreen.svg)]()

This repository contains our code to answer the Machine Learning class Homework 4.
We will train agent to play **Grid World** by using the **Monte Carlo** algorithm.
Our code is a modification based on the code available from the
[RLCode Github](https://github.com/rlcode/reinforcement-learning/tree/master/1-grid-world/3-monte-carlo)

## How to run

You can run the program by executing the following commands:

```shell
cd YOUR_DIR
python mc_agent.py i # to run scenario 1 (normal)
python mc_agent.py ii # to run scenario 2 (add one more triangle)
```

## Code Breakdown

We are going to briefly explain some of the important parts of the code that is related to the
Monte Carlo algorithm in general.

### Agent's Global Variables

```python
self.actions = actions
self.learning_rate = 0.01
self.discount_factor = 0.9
self.epsilon = 0.1
self.samples = []
self.value_table = defaultdict(float)
```

The `actions` are the list of available actions from the grid world, which is `['u', 'd', 'l', 'r']` up, down, left, 
right respectively. The `learning rate` and `discount_factor` are the parameters to update the `value_table`. The 
`epsilon` is used for the epsilon greedy approach. The `samples` are used to temporarily stored the value for each 
state in a episode. Lastly the `value_table` is initialized with the number of `0.0`.

### Epsilon Greedy -- Determine the action to the next state

```python
def get_action(self, state):
    if np.random.rand() < self.epsilon:
        # take random action
        action = np.random.choice(self.actions)
    else:
        # take action according to the q function table
        next_state = self.possible_next_state(state)
        action = self.arg_max(next_state)
    return int(action)
```

There are two things that may happen when an agent proceed to the `next_state` in an episode:
* **Exploration**. The epsilon greedy approach takes a `random` number and compare it with the `epsilon`. If the 
`random` is smaller that`epsilon` then the agent will pick one action randomly.
* **Exploitation**. The agent will look up for the `next_state` from the environment and get the `value_table` for
the `next_state`. Then, the agent will pick the action that has the 
`maximum value`. If multiple action with the same maximum value exists, the agent will pick a random action based on 
those available maximum actions.

So, the role of `epsilon` in this example is to give the agent chance to explore the environment (10% chance in this 
example). If the `epsilon` is extremely low (or set to 0) than the agent will have less chance (or no chance) to 
explore and the `value_table` will most likely to have `bias` from the previous episodes. Thus, we cannot see many
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

### Updating the value_table when one episode ends

```python
# at the end of each episode, update the q function table
if done:
    print("episode : ", episode)
    agent.update()
    agent.samples.clear()
    break
```

Once the agent reaches the ending of an episode either the circle grid or the triangle grid, the agent can
start running the update for q value function to improve the value in the `value_table` (Remember that at 
the beginning the `value_table` is initialized with zeros)

```python
# for every episode, agent updates q function of visited states
def update(self):
    G_t = 0
    visit_state = []
    for reward in reversed(self.samples):
        state = str(reward[0])
        if state not in visit_state:
            visit_state.append(state)
            G_t = self.discount_factor * (reward[1] + G_t)
            value = self.value_table[state]
            self.value_table[state] = (value + self.learning_rate * (G_t - value))
```

When the agent moves from one state to the next state, it records the state and reward information to the
`samples` list. At the end of the episode, the agent will reflect to the `samples` info and start updating
the value. There is one caveat here, the agent only update one value from each state. For instance, if the
agent moves from grid `(1,1)` to `(2,1)` to `(3,1)` and back to `(2,1)`, the agent can only update the value
in the grid `(2,1)` once even though it moves through this state twice.

The value for the state will be updated following the Monte Carlo equation as follows:
```python
G_t = self.discount_factor * (reward[1] + G_t)
value = self.value_table[state]
self.value_table[state] = (value + self.learning_rate * (G_t - value))
```
