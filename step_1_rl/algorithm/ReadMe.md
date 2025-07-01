## RL Algorithms for Continuous Action Spaces

### (1) Policy Gradient Algorithms
|Algorithm|Explanation|
|---------|-------------------------|
|1. TRPO (Trust Region Policy Optimization)|Policy Update를 할 때에 old policy와 new policy간의 KL divergence의 제한을 걸어 안정성을 보장한다.|
|2. PPO (Proximal Policy Optimization)|TRPO 알고리즘을 기반으로 하는데, policy gradient를 최적화하기 위해서 constraint lower bound를 사용하기 보다는 clipping을 사용한다.|
|3. DDPG (Deep Deterministic Policy Gradient)|Actor-Critic 기반의 알고리즘으로, Critic이 Q-value를 예측하면 Actor이 deterministic action을 뱉는다.|
|4. TD3 (Twin Delayed Deep Deterministic Policy Gradient)|DDPG의 개선된 버전의 알고리즘으로, overestimation 문제를 완화하기 위해서 2개의 Critic network를 사용하며, policy update를 지연시켜 학습을 더 안정화 한다.|
|5. SAC (Soft Actor Critic)|Max entropy framework를 사용하여 행동의 다양성을 유지한다.|


### (2) Q-learning Algorithms
|Algorithm|Explanation|
|---------|-------------------------|
|1. CEM-RL (Cross-Entropy Method Reinforcement Learning)|Continuous action space를 sampling 기반의 최적화로 해결하고, value function과 결합해 동작한다.|
|2. QT-Opt (Q-Learning for Continuous Control)|Q-learning 기반으로 continuous action space에서 동작하고, 행동을 샘플링하여 최적의 행동을 탐색한다.|


### (3)두 계열 중 어떤 알고리즘을?
- 일단 race driving은 action space와 state space가 모두 "continuous" 해야 한다.
    - Policy gradient 기반의 알고리즘들은 행동 분포(e.g Gaussian distribution)를 직접 모델링하기 때문에 continuous action space에서 더 자연스럽게 동작이 가능하다.
    - 하지만 Q-learning 기반의 알고리즘들은 모든 (state, action) pair들에 대해서 Q-value estimation을 해야 하기 때문에 continuous action space를 처리하기에는 계산 비용이 너무 높다.

- 또한, continuous action space는 deterministic 하지 않다.
    - 즉, 어떤 state에서 하나의 optimal action만이 항상 존재하기 어려울 수 있으며, 행동에 noise가 추가될 가능성이 있다는 뜻이다.
    - 이런 경우, policy gradient 기반의 알고리즘들이 stochastic policy를 학습하기 때문에 noise가 포함된 상황에서 더 유연하게 대처가 가능하다.

- 정책을 학습함에 있어서 trajectory sample을 모아야 하기 때문에 "Exploration"이 중요하다.
    - Policy gradient 기반의 알고리즘은 stochastic policy를 사용하기 때문에 자연스럽게 exploration이 가능하다.
    - 하지만, Q-learning 기반의 알고리즘들은 $\epsilon$-greedy로 탐험을 하는데, 이는 당연하게도 stochastic policy를 학습하는 것에 비해서 덜 효율적일 것이다.

- Better optimization
    - Policy gradient 기반의 알고리즘은 entropy normalization(SAC), KL divergence(TRPO), Clipping(PPO) 등과 같은 다양한 기법으로 안정성과 샘플의 효율성을 높인다.
    - 반면, Q-learning 기반의 알고리즘들은 학습 중 발생 가능한 instability 문제를 해결함에 있어서 어려운 경우가 있다.



## 벨만 방정식 (가치 함수의 재귀적 성질)
$\to$ 벨만 방정식을 사용하여 상태 가치 함수 ($V(s)$), 행동 가치 함수 ($Q(s,a)$)를 재귀적으로 표현할 수 있다.  
- 가치 함수 하는 것은 **정책의 성능**을 평가하기 위한 중요한 지표이다.
- 하지만 가치 함수를 직접 계산하는 것이 intractable하기 때문에 가치 함수를 **추정**해야 한다.
     - 이 때 사용되는 것이 벨만 방정식이다.
     - **즉, 실제 가치 함수라면 벨만 방정식을 만족해야 하기 때문에, 정의된 벨만 방정식의 좌변과 우변의 차이가 0이 되도록 $V_{\theta}(s)$, 또는 $Q_{\theta}(s,a)$를 학습 시키게 되는 것이다.**


### (1) Return의 재귀적 표현
$G_{t} \\\ = R_{t} + \gamma R_{t+1} + \gamma^2 R_{t+2} + ... \\\  = R_{t} + \gamma G_{t+1}$  


$\to$ 위와 같이 나타내면 1-step return이라고 하며, 즉각적인 보상을 몇번까지 사용하느냐에 따라서 n-step return으로 나타낼 수 있다.

### (2) 상태 가치 함수의 재귀적 표현
$V^{\pi}(s) \\\  = E_{\pi}[R_{t} + \gamma V^{\pi}(S_{t+1})|S_{t} = s] \\\ = \sum_{a\in A} \pi(a|s) [r(s,a) + \gamma \sum_{s' \in S} p(s'|s,a)V^{\pi}(s')]$  

$\to$ 이 식을 상태 가치 함수에 대한 Bellman Equation이라고 한다.

### (3) 행동 가치 함수의 재귀적 표현
$Q^{\pi}(s,a) \\\ =E_{\pi} [G_t|S_t=s,A_t=a] \\\ = r(s,a) + \gamma \sum_{s' \in S} P(s'|s,a)\sum_{a'\in A} \pi(a'|s') Q^{\pi} (s',a')$  

$\to$ 이 식을 행동 가치 함수에 대한 Bellman Equation이라고 한다.
- 앞선 상태 가치 함수와 달리 $r(s,a)$를 상수 값으로 뺄 수 있는 이유는 action값 역시 정해져 있기 때문에 $R(s,A_{t})$에서의 action 확률변수가 없기 때문이다.


## 가치 함수 근사하기
### (1) Stochastic Approximation
- 어떤 확률 변수의 기댓값을 알 수 없을 때, **표본 평균**을 사용해서 실제 평균을 추정하곤 한다.
- 

### (2) Monte-Carlo Evaluation


### (3) Temporal Difference Evaluation