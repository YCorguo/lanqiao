# AI Note

## Reinforcement Learning

### environment of RL

$next\_state = P(\cdot|recent\_state, action\_of\_agent)$

### multi-armed bandit problem

#### problem description
There is one K-armed bandit. Pulling each lever corresponds to a probability distribution of rewards. We start from scratch with the unknown probability distribution of rewards for each lever, aiming to obtain the highest possible cumulative reward after operating $T$ times.

#### format desctiption
$states: <A, R>$
$A$ is the action set(multiset), and the action space is $\{a_1, \ldots, a_K\}$
$R$ each level corresponds to one $R_i(r|a)$
$target:\max\sum_{t=1}^Tr_t,\ s.t.\ r_t\thicksim R(\cdot|a_t)$

### two types of RL

#### model-based reinforcement learning

#### model-free reinforcement learning

### classic RL algorithms

#### DQN algo.

#### PPO algo.

##### optimization target of TRPO algo.

$$\max\limits_\theta \Bbb{E}_{s\thicksim\nu^{\pi_{\theta_k}}}\Bbb{E}_{a\thicksim\pi_{\theta_k}}(\cdot|s)[\frac{\pi_\theta(a|s)}{\pi_{\theta_k}(a|s)}A^{\pi_{\theta_k}}(s,a)]\\
s.t.\ \Bbb{E}_{s\thicksim\nu^{\pi_{\theta_k}}}[D_{KL}(\pi_{\theta_k}(\cdot|s), \pi_\theta(\cdot|s))]\le\delta
$$



## references

[动手学强化学习](https://hrl.boyuai.com/)