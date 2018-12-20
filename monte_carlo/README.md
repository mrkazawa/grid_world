# Machine Learning HW 4 - Monte Carlo

[![build](https://img.shields.io/badge/build-pass-green.svg)]()
[![code](https://img.shields.io/badge/code-python3.5-yellowgreen.svg)]()

This repository contains our code to answer the Machine Learning class Homework 4.
We will train an agent to play **Grid World** by using the **Monte Carlo** algorithm.
Our code is a modification based on the code available from the
[RLCode Github](https://github.com/rlcode/reinforcement-learning/tree/master/1-grid-world/3-monte-carlo)

## How to run

You can run the program by executing the following commands:

```shell
cd YOUR_DIR
python mc_agent.py i # to run scenario 1 (normal)
python mc_agent.py ii # to run scenario 2 (decreasing epsilon)
python mc_agent.py iii # to run scenario 3 (add one more triangle)
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
`random` is smaller that `epsilon` then the agent will pick one action randomly.
* **Exploitation**. The agent will look up for the `next_state` from the environment and get the `value_table` for
the `next_state`. Then, the agent will pick the action that has the
`maximum value`. If multiple actions with the same maximum value exists, the agent will pick a random action based on
those available maximum actions.

So, the role of `epsilon` in this example is to give the agent chance to explore the environment (10% chance in this
example). If the `epsilon` is extremely low (or set to 0) then the agent will have less chance (or no chance) to
explore and the `value_table` will most likely to have `bias` from the previous episodes. Thus, we cannot see any
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

### Our Modification - Calculate the mean loss and plot the chart

Inspired by the neural network, we want to modify the code to calculate the `mean loss` per episode.
First, we calculate the loss per grid that the agent visits by using the following code

```python
# append normalized losses to the list
losses.append(abs(G_t - value) / 100)
```

At the end of each episode, we calculate the mean of those losses that the agent visits.
We store this mean per episode to the `history` object.

```python
# save the loss to history
history["losses"].append(np.mean(losses))
```

Finally, when all of the episodes end, we plot the `history`.

```python
def plot_loss(history):
    f, ax = plt.subplots(figsize=(7, 4))
    f.canvas.set_window_title('Result')

    ax.set_title('Training Loss')
    ax.set_xlabel('episodes')

    loss = history['losses']
    episodes = range(len(loss))

    ax.plot(episodes, loss, 'r', label='loss')

    ax.legend()
    plt.tight_layout()
    plt.show()
```

## Scenario 1 - Running the original code

In this scenario, we are going to run the original code from the
[RLCode Github](https://github.com/rlcode/reinforcement-learning/tree/master/1-grid-world/3-monte-carlo).

First Stage | Last Stage
:---: | :---:
![first](results/scenario1/env.png?raw=true "first") | ![last](results/scenario1/heat_map.png?raw=true "last")

At the left figure, we can see the initialization stage of the program. The value for each grid in the
picture will be set to `0.0`. This value will be updated after each episode as the agent learning to
reach the `circle`.

The figure in the right shows the updated `value_table` after `500 episodes`. The value where the `triangle`
resides has minus value because the reward is `-100` and we do not want the agent to fall to it.
Assuming the starting point is grid `(1,1)`, the agent moves from grid `(1,1)` to `(2,1)` to `(3,1)` to `(4,1)` to
`(4,2)` to `(4,3)` and finally `(3,3)`. This is the path that the agent has explored and learned during
the training. We can see that this path contains grids with higher values. When following the greedy policy,
the agent will choose these grids over the others and eventually reaches the `circle`.

> The grids with the value of 0 means that during the training, the agent never visits those grids.


![loss](results/scenario1/loss.png?raw=true "loss")

The training loss is depicted at the figure above. In the early episodes the losses were high because the
agent could not get the prediction value number for each grid correctly. However, as the episodes continue,
we can clearly see that the agent is getting better at predicting the value and generating a smaller number of losses.
In the figure, we can see several spikes occurred during the training. This spikes happened when
the agent chooses to **explore** instead of **exploit** (Remember we use epsilon `0.1`, therefore, we have `10%`
chance to explore even though at the later episodes, it should not need to explore anymore)

## Scenario 2 - Decreasing epsilon as the episodes continue

In this scenario, we are going to run a simple epsilon modification to the original code.
Our lemma is the following:

> The agent should do less exploration as he gets better in predicting the value

Therefore we decrease the epsilon value as the episodes increase. When it reaches 100 episodes,
the epsilon will be set to `0.075` and then decreased to `0.05` at 200 episodes, `0.025` at 300 episodes.
Finally, we don't use epsilon (full greedy) at the last 100 episodes.

```python
# our custom decreasing epsilon trial
if scenario == "ii":
    if episode == 100:
        agent.epsilon = 0.075
    elif episode == 200:
        agent.epsilon = 0.05
    elif episode == 300:
        agent.epsilon = 0.025
    elif episode == 400:
        agent.epsilon = 0
```

First Stage | Last Stage
:---: | :---:
![first](results/scenario2/env.png?raw=true "first") | ![last](results/scenario2/heat_map.png?raw=true "last")

The figure in the right shows the updated `value_table` after `500 episodes`. Assuming the starting point is grid `(1,1)`,
the agent moves from grid `(1,1)` to `(1,2)` to `(1,3)` to `(1,4)` to `(2,4)` to
`(3,4)` and finally `(3,3)`. This is the path that the agent has explored and learned during
the training. We can see that this path contains grids with higher values. This path is **different** from
the path in *scenario 1*. Thus, we can conclude that the agent can choose one among all of the possible scenarios.

![loss](results/scenario2/loss.png?raw=true "loss")

In our previous scenario, we state that `the spike happens in the loss chart is due to the epsilon value`.
In this scenario, we can surely confirm that statement. We can see from the figure that spikes still happen however
the frequency of the spike decreases as the episodes continue. This happens as the agent has lesser chance to explore
in the later episodes. After 400 episodes, we can see that we cannot find any spikes in the chart because at that moment
the agent is at `full greedy` or `full exploit` mode.

## Scenario 3 - Add one more triangle

In this last scenario, we introduce one new triangle to the environment. We put the location of the new triangle very close to the
starting location at grid `(2,2)`. Our original intention in choosing this location is to make the agent difficult to reach
the `circle` by continuously falling to the trap when he explores. However, the agent can still be able to reach the destination.

First Stage | Last Stage
:---: | :---:
![first](results/scenario3/env.png?raw=true "first") | ![last](results/scenario3/heat_map.png?raw=true "last")

The figure in the right shows the updated `value_table` after `500 episodes`. Assuming the starting point is grid `(1,1)`,
the agent moves from grid `(1,1)` to `(2,1)` to `(3,1)` to `(4,1)` to `(5,1)` to `(5,2)` to `(5,3)` to
`(4,3)` and finally `(3,3)`. This is the path that the agent has explored and learned during
the training. We can see that this path contains grids with higher values. This path is **different** from
the path in *scenario 1* and *scenario 2*.

This environment originally can be solved by 6 moves. However, in this scenario, the agent solved it in 8 moves.
The solution depends on the exploration. From grid `(4,1)` the agent has two choices: move to grid `(5,1)` or `(4,2)`.
Based on our training, the agent discovered the `circle` when he moved to `(5,1)`.
Therefore the value in `(5,1)` updated sooner than `(4,2)`. When grid `(5,1)` receives more visit by
the agent over multiple episodes, the value for `(5,1)` will keep increasing and eventually, grid `(4,2)` will be `ignored`.

![loss](results/scenario3/loss.png?raw=true "loss")

The figure depicts the loss for this scenario. However, there is nothing more significant can be found from this figure
because it is relatively similar to the one in *scenario 1*.