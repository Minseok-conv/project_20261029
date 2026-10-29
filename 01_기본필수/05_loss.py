"""실습 05: 정답 인덱스와 logits로 손실을 계산합니다."""
import torch
from torch import nn

logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
target = torch.tensor([0, 1], dtype=torch.long)
loss_fn = nn.CrossEntropyLoss()
# nn.CrossEntropyLosss()는 내부적으로 logits에 필요한 변환을 적용해서 cross entropy를 계산함.
# 따라서 softmax()를 적용해서 확률로 만들어줄 필요없이 logits값을 그대로 CrossEntropyLoss에 전달함
loss = loss_fn(logits, target)

print("logits:", tuple(logits.shape))
print("target:", tuple(target.shape), target.dtype)
print("loss:", round(loss.item(), 6))
