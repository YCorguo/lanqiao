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

#### cumulative regret
For each action $a$, define expected reward as $Q(a)=\Bbb{E}_{r\thicksim R(\cdot|a)}[r]$. Therefore, $\exist Q^* = \max_{a\isin A}Q(a)$. Intuitively, the regret of the action $R(a) = Q^* - Q(a)$, and cumulative regret, for next complete T steps, is $\sigma_R = \sum_{t=1}^TR(a_t)$.

Further, the inference below allow us to dynamically renew expected rewards:
$$\begin{split}Q_k&=\frac{1}{k}\sum\limits_{i=1}^{k}r_i\\
      &=\frac{1}{k}\sum\limits_{i=1}^{k-1}r_i+\frac{r_k}{k}\\
      &=\frac{k-1}{k}(\frac{1}{k-1}\sum\limits_{i=1}^{k-1}r_i)+\frac{r_k}{k}\\
      &=\frac{k-1}{k}Q_{k-1}+\frac{r_k}{k}\\
      &=Q_{k-1}-\frac{1}{k}Q_{k-1}+\frac{r_k}{k}\\
      &=Q_{k-1}+\frac{r_k-Q_{k-1}}{k}\\
\end{split}$$

For each lever, only if we use a counter $N(a)$, updates of $\^Q(a_t)$ could be descripted as:
- for $\forall a \isin A,\ init.\ N(a) = \^Q(a) = 0$
- for $t = 1 \to T$ do
    - choose lever $a_t$
    - get $r_t$
    - update counter: $N(a_t) = N(a_t) + 1$
    - update expected rewards: $\^Q(a_t)=\^Q(a_t)+\frac{1}{N(a_t)}[r_t-\^Q(a_t)]$
    - end for

#### $\epsilon$-Greedy algo.

Optimize the lever choosing strategy, balancing exploration and exploitation. The choosing strategy is:
$$a_t = \left\{\begin{aligned}\arg \max_{a\isin A}\^Q(a), with\ prob.\ 1-\epsilon\\random\ sample\ from\ A, with\ prob.\ \epsilon\\\end{aligned}\right.$$

If set $\epsilon$ as a constant, the cumulative regret will linearly increase. But if set $\epsilon = \frac{1}{t}$, it will be sublinear and obviously better than constant form.

#### Upper Confidence Bound algo.
##### Hoeffding's inequality
Given n i.i.d random variables $X_1, X_2, \ldots, X_n$, with a range of $[0, 1]$, experience expectations is $\overline x_n = \frac{1}{n}\sum_{j=1}^nX_j$, then$$\Bbb{P}\{\Bbb E[X]\ge\overline x_n+u\}\le e^{-2nu^2}$$

##### UCB in MAB for each lever
let $\overline x_t = \^Q_t(a), u = \^U_t(a), p = e^{-2N_t(a)U_t(a)^2}$, then $$\Bbb{P}\{Q_t(a)\ge\^Q_t(a)+\^U_t(a)\}\le e^{-2n\^U^2_t(a)} = p$$
$$1-\Bbb{P}\{Q_t(a)\ge\^Q_t(a)+\^U_t(a)\}\ge 1-p$$
$$\Bbb{P}\{Q_t(a)<\^Q_t(a)+\^U_t(a)\}\ge 1- p$$
when $N_t$ increases, $p$ is decreases. so $Q_t(a) = \^Q_t(a)+\^U_t(a)$, and $\^Q_t(a)+\^U_t(a)$ is expected reward upper bound. Now, we can choose the action with the reward expectation with largest upper bound.

Using $\epsilon$-greedy algo., we could set $p = \epsilon = \frac{1}{t}$, and because $p = e^{-2N_t(a)U_t(a)^2}$, get $\^U_t(a) = \sqrt{\frac{-\log p}{2N_t(a)}}$, and of course, for robustness, $\^U_t(a) = \sqrt{\frac{-\log p}{2(N_t(a)+1)}}$.

At last, we could set a coefficent $c$ to control the weight of uncertainty: $$a = \arg \max _{a \isin A} [\^Q_t(a)+c \cdot \^U_t(a)]$$.

#### Thompson Sampling
1. assume that each lever corresponds to one specific distribution.
2. sample on each lever to estimate the specific distribution.
3. choose the action of largest reward.

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

## Fundamental

### Cross Validation
1. Simple Cross Validation.
Simply split dataset as train set and test set.
2. K-fold Cross Validation
    a. split
    b. enumerate each subset as validset.
    c. use mean value of K times.
3. Leave-one-out Cross Validation
    a. for N samples, use N-1 to train, 1 for evaluate.
4. Usually, if the size is large, simply split it into 10 or 20 parts, else use Sturge's Rule:$$Number\ of\ Bins=1+log_2(N)$$

### Measurements
$$Accuracy = \frac{TP+TN}{TP+TN+FP+FN}$$
$$Precision = \frac{TP}{TP+FP}$$
$$Recall = TPR = \frac{TP}{TP+FN}$$
$$FPR = \frac{FP}{TN+FP}$$
$$F1 = \frac{2*Precision*Recall}{Precision+Recall}$$
$$ROC\ Curve = \left\{
\begin{aligned}
x: FPR \\
y: TPR \\
\end{aligned}
\right.
$$
$$LogLoss = -1.0\times(target\times log(prediction)+(1-target)\times log(1-prediction))$$
$$Macro\ averaged\ precision = \frac{\sum\limits_i Precision_i\times N_i}{\sum\limits_i N_i}$$
$$Micro\ averaged\ precision = \frac{Precision_i}{N_i}$$
$$Weighted\ averaged\ precision = \frac{\sum\limits_i Precision_i\times N_i\times w_i}{\sum\limits_i N_i}$$
$$Confusion\ Matrix:
\begin{matrix} 
\ & class-1 & class-0 \\
class-1 & xx & xx \\
class-0 & xx & xx \\
\end{matrix}$$
$$Error=True Value−Predicted Value$$
$$Absolute Error=Abs(True Value−Predicted Value)$$
$$Squared Error=(TrueValue−Predicted Value)^2$$
$$RMSE=SQRT(MSE)$$
$$Percentage Error=\frac{True Value–Predicted Value}{True Value}\times100$$
$$Coefficient\ of\ determination = R^2 = \frac{\sum_{i=1}^N(y_{t_i}-y_{p_i})^2}{\sum_{i=1}^{N}(y_{t_i}-y_{t_{mean}})}$$
$$MCC=\frac{TP\times TN−FP\times FN}{\sqrt{(TP+FP)\times(FN+TN)\times(FP+TN)\times(TP+FN)}}$$
## references

[动手学强化学习](https://hrl.boyuai.com/)






















