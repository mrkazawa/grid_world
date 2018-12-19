# Machine Learning HW 4 - Q Learning

[![build](https://img.shields.io/badge/build-pass-green.svg)]()
[![code](https://img.shields.io/badge/code-python3.5-yellowgreen.svg)]()

This repository contains our code to answer the Machine Learning class Homework 4.
We will train agent to play **Grid World** by using the **Q Learning** algorithm.
Our code is a modification based on the code available from the
[RLCode Github](https://github.com/rlcode/reinforcement-learning/tree/master/1-grid-world/5-q-learning)

## How to run

You can run the program by executing the following commands:

```shell
cd YOUR_DIR
python mc_agent.py i # to run scenario 1 (normal)
python mc_agent.py ii # to run scenario 2 (decreasing epsilon)
python mc_agent.py iii # to run scenario 3 (add one more triangle)
python mc_agent.py iv # to run scenario 4 (cliff walking env)
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
action = agent.get_action(str(state))
next_state, reward, done = env.step(action)

# with sample <s,a,r,s'>, agent learns new q function
agent.learn(str(state), action, reward, str(next_state))

state = next_state
```

First the agent has to get the `current state (S)` and the `action (A)` from the current state.
Then, it does the action and get the `reward (R)` and the `next state (S')`. By using all of those information,
the agent can update the `q_table`. After the value is updated, the `next state (S')` will be the `current state (S)`.
Then, the program continue following the same logic.

The policy in `q_table` will be updated following the Q Learning equation as follows:

```python
# update q function with sample <s, a, r, s'>
def learn(self, state, action, reward, next_state):
    current_q = self.q_table[state][action]
    # using Bellman Optimality Equation to update q function
    new_q = reward + self.discount_factor * max(self.q_table[next_state])
    self.q_table[state][action] += self.learning_rate * (new_q - current_q)
```

The difference between
[SARSA](https://github.com/mrkazawa/grid_world/tree/master/sarsa)
and Q Learning can be found in the action for the next state `(A')`

* In **SARSA**, we include the action for the next state `(A')` to update the `q_table`. This
action can be random (when the algorithm to determine next state hit the `epsilon`) or
it can be greedy (looking for the max action in the `q_table` for the next state)
* In **Q Learning**, the action for the next state `(A')` is not included in the `q_table` update.
Instead, it uses the greedy approach to look for maximum policy in `q_table` for the next state.


