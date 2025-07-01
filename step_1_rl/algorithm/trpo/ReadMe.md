## TRPO (Trust Region Policy Optimization)
### 0. Policy Gradient Methods
- 강화학습에서 "정책"이라는 것은 주어진 상태에서 어떤 행동을 취할지 알려주는 것으로, 결국 주어진 상태 $s$에서 어떤 행동 $a$를 선택할 조건부 확률이다.
- Parameterized function으로 이 정책을 모델링하여 optimal policy를 찾는 방법을 **Policy-Based** 방법이라고 한다.
- 얻은 정책의 가치를 평가할 수 있게 해주는 성능 지표를 $J(\theta)$라고 한다면 우리는 이 함수를 최대화 하는 방향으로 $\theta_{new} = \theta_{old} + \alpha \triangledown J(\theta_{old})$와 같이 파라미터를 업데이트 할 것이고, 이 방법을 **Policy Gradient**라고 한다.  

$\triangledown_{\theta}J(\theta)  \propto \sum_{s\in S}d_{\pi_{\theta}}(s)\sum_{a\in A|Q^{\pi_{\theta}}}(s,a) \triangledown_{\theta} \pi_{\theta} (a|s)$  

= $\sum_{s \in S} d_{\pi_{\theta}}(s)\sum_{a\in A} Q^{\pi_{\theta}}(s,a)\pi_{\theta}(a|s) \frac{\triangledown_{\theta}\pi_{\theta}(a|s)}{{\pi_{\theta}}(a|s)}$  

= $E_{\pi_{\theta}}[Q^{\pi_{\theta}}(s,a) \triangledown_{\theta}log \pi_{\theta}(a|s)]$  

- 위의 식에서 $\triangledown_{\theta}J(\theta)$와 등호를 이루는게 아니라 비례식으로 쓰이고 있음을 인지하고 있어야 한다.

    - 어차피 $J(\theta)$의 $\theta$에 대한 gradient를 그대로 정확히 구하는 것은 불가능하지만, 기댓값이라면 Monte Carlo 기법을 사용해 근사를 할 수 있다.
    - 정책 $\pi_{\theta}$로 환경과 상호작용을 많이 해서 trajectory를 생성하여 각각의 $(s_t, a_t)$마다 $Q^{\pi_{\theta}}(s,a)\triangledown_{\theta}log\pi_{\theta}(a|s)$를 계산해서 표본 평균을 구하면 되는 것이기 때문이다.
    - 결국에 **$Q^{\pi_{\theta}}(s,a)$를 어떤 값을 대체하여 구하는지에 따라서** policy gradient의 알고리즘들이 다양하게 되는 것이다.

- $d_{\pi_\theta}(s) = \sum^{\infin}_{t=0}\gamma^tPr(s_t=s|\pi_{\theta})$ 이고, $Pr(s_t=s|\pi_{\theta})$ 는 정책 $\pi_\theta$를 따를 때  $t$시점에서의 상태가 $s$일 확률을 의미한다.


### 1. `REINFORCE (ON-POLICY)`
- 알고리즘 개요
1. 하나의 episode를 진행하여 하나의 trajectory ($\tau = (s_0,a_0,r_0,s_1,a_1,r_1,...,s_T)$)를 얻음
2. policy gradient의 추정치 (위에서 얻은 식)을 사용해서 gradient ascent를 진행.
3. `n_episode`만큼 위의 1, 2 단계를 반복하며, 한번의 episode가 끝나면 replay buffer을 모두 비워준다. (이는 on-policy method이기 때문이다.)

- 특징
1. `REINFORCE`는 policy optimization method의 시초이고, policy의 가치를 나타낼 수 있는 함수로 제일 간단한 $R_t(\tau)$를 사용한다.
    - MonteCarlo estimate로 생성된 하나의 궤적을 이용해 policy gradient ($\triangledown_{\theta}(J(\theta))$)를 추정한다는 의미이다.
    - 이는 **bias**는 없지만, 추정값이 **큰 variance**를 갖는다는 단점이 있다.
    - 따라서 더 향상된 `REINFORCE` 알고리즘으로 추정량의 분산을 줄이기 위해서 **Baseline**을 많이 사용한다.

2. `REINFORCE with Baseline`
    - $\triangledown_{\theta}J(\theta) \propto \sum_{s\in S}d_{\pi_{\theta}}(s)\sum_{a\in A}(Q^{\pi_{\theta}}(s,a) - b(s))\triangledown_{\theta}\pi_{\theta}(a|s)$
    - 위의 식에서의 $b(s)$가 baseline으로서의 역할을 한다.
    - 보통 baseline으로는 action의 영향을 받지 않는 함수를 사용하게 되는데, 때문에 가치 함수인 $V^{\pi_{\theta}}(s)$를 많이 사용한다.
    - 당연하겠지만 우리가 직접 상태 가치 함수를 구하기란 쉽지 않다. 역시나 매개변수화를 통해서 상태 가치 함수를 모델링해야 한다는 뜻이다.


3. `REINFORCE with Baseline` vs `Actor-Critic`
    - `Actor-Critic`에 대해서는 아직 언급하지 않았으나, 표면적으로 볼 때는 두 알고리즘이 명칭만 다르게 할 뿐, `critic`이라는 가치 함수 추정하는 network를 추가하였다는 점에서 동일해 보였다.
    - 하지만 차이라고 하면, 언제 `critic`의 파라미터를 업데이트 하는지에 대한 차이가 있었다.
    - `Actor-Critic`은 bootstrapping 방법을 쓰기 때문에 매 step마다 advantage를 예측하는 `critic`의 파라미터가 업데이트 된다.
    - 하지만, `REINFORCE with Baseline`은 monte carlo 방법을 쓰기 때문에 하나의 episode가 끝날 때마다 `critic`의 파라미터를 업데이트 한다.