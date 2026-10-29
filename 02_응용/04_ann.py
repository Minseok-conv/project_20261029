"""실습 04: 은닉 너비에 따라 파라미터 수가 변합니다. 학습 성능 비교는 아닙니다."""
import torch
from torch import nn

torch.manual_seed(42)
#seed를 정하면 맨 처음 모델 생성시 가중치에 랜덤으로 정하는 값을 고정할 수 있다. 
X = torch.ones(4, 2)
for hidden in [4, 8, 16]:
    model = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(),  nn.Linear(hidden, 2))
    logits = model(X)
    count = sum(p.numel() for p in model.parameters())
    print("은닉:", hidden, "출력:", tuple(logits.shape), "파라미터:", count)
