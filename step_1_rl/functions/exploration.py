'''
[Possible Methods for Exploration]
1. Noisy Networks for Exploration

2. Boltzman Exploration
- SAC는 기본적으로 Gaussian Distribution을 따르는 확률적 정책이다.
- 각 행동에 대한 확률을 temperature을 조정하여 설정할 수 있는데, 이 값을 낮추면 탐험이 줄어들고, 높은 값을 설정하면 탐험이 증가한다.

3. Intrinsic Motivation (Curiosity-Driven Exploration)
- Agent가 예상하지 못한 상황이나 변화를 경험할 때 보상을 받도록 설계하여 새로운 상태를 탐색하도록 유도
- Intrinsic Reward를 정의해야 함.

4. Randomized Data Augmentation
- 데이터 환경, 결국 track 환경을 다양하게 구현해야 함.

5. Entropy Regularization
- SAC는 자체적으로 entropy 보상을 포함해서 행동의 불확실성을 높이는 알고리즘이다.
- entropy 항목을 조절해서 exploration / exploitation의 균형을 조절 가능.
'''