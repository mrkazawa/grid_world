



https://github.com/rlcode/reinforcement-learning/tree/master/1-grid-world/5-q-learning


The difference between SARSA and Q Learning can be found in the action for the next state
* In SARSA, the action for the next state is given by the environment. In this case,
the action can be random (when the algorithm to determine next state hit the `epsilon`) or
it can be greedy (looking for the max action in the `q_table`)
* In Q Learning, the action for the next state is always determined by `q_table`.
Therefore, it always be greedy (looking for the max action in the table)