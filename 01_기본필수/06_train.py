"""실습 06: 작은 정상 자료로 학습 순서만 익힙니다. 여기는 train 오차 확인용입니다."""
import torch
from torch import nn

torch.manual_seed(42)
X = torch.tensor([[-2., -1.], [-1., -1.], [-1., 0.], [0., -1.],
                  [1., 0.], [0., 1.], [1., 1.], [2., 1.]])
y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
print("학습 전 loss:", round(loss_fn(model(X), y).item(), 6))
model.train()
# model을 학습용으로 전환 -> 예를 들어 이 상태에서는 Dropout이 작동함 
for epoch in range(80):
    optimizer.zero_grad()
    loss = loss_fn(model(X), y)
    loss.backward()
    optimizer.step()
model.eval()
# model을 평가용으로 전환 -> 이 상태에서는 Dropout같은 것이 작동 안함.
with torch.no_grad():
    logits = model(X)
    print("학습 후 loss:", round(loss_fn(logits, y).item(), 6))
# item()은 Tensor객체를 python 숫자로 변환함. 
# item()은 Tensor가 가진 원소 개수가 1개일때만 사용가능함. 
    print("train 정확도:", round((logits.argmax(dim=1) == y).float().mean().item(), 4))
print("주의: 위 정확도는 학습한 8건의 결과이며 새로운 자료의 성능이 아닙니다.")

#  PyTorch의 autograd는 Tensor 연산이 실제로 실행될 때 그 연산들을 추적하면서 계산 그래프를 동적으로 만든다.
#  requires_grad = True 인 Tensor와 관련된 연산들 즉, gradient가 필요한 연산을 추적해서 계산그래프를 만들고 해당 연산관계를 기록함