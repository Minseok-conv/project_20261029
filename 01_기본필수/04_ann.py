"""실습 04: 2 → 8 → 2 신경망의 순전파를 확인합니다."""
import torch
from torch import nn

torch.manual_seed(42)

#  Linear의 weight와 bias는 처음에 임의의 값으로 초기화되는데, 
#  seed를 고정하면 실행할 때마다 동일한 초기값에서 학습을 시작할 수 있음.
model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
#  nn.Linear(2,8) 입력값 2개를 받아서 출력값 8개를 출력하는 완전 연결층.
#  nn.Linear(2,8) 의 경우 bias 8개, weight 16개, nn.Linear(8,2)의 경우 bias 2개, weight 16개 -> 총 파라미터는 42개임.
X = torch.tensor([[1.0, 2.0], [3.0, 4.0], [0.0, 1.0], [2.0, 0.0]])
logits = model(X)
print("입력:", tuple(X.shape))
print("출력:", tuple(logits.shape))
print("파라미터 수:", sum(p.numel() for p in model.parameters()))

# logits는 원시 점수입니다. 이 시점에는 학습하지 않았습니다.
# model.parameters()의 결과로 대략 4개의 parameter가 차례대로 나옴
# for p in model.parameters():
#    print(p.shape)

# 첫 번째 Linear의 weight → shape [8, 2]
# 첫 번째 Linear의 bias   → shape [8]
# 
# 두 번째 Linear의 weight → shape [2, 8]
# 두 번째 Linear의 bias   → shape [2]

# numel()은 해당 tensor 내부의 element 개수를 센다.
# 정확히는 여기서 p는 Tensor객체가 아니라 torch.nn.Parameter객체이다.
# Parameter객체는 PyTorch에게 이 Tensor는 모델이 학습해야 할 parameter임을 알려주는 특별한 Tensor이다. 