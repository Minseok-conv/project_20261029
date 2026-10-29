"""실습 05: 마지막 작은 batch를 버리지 않고 자료를 모두 읽습니다."""
from pathlib import Path
import csv
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

X = torch.arange(20, dtype=torch.float32).reshape(10, 2) / 10
y = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long)
dataset = TensorDataset(X, y)
# TensorDataset(X,y)는 X[i]와 y[i]를 한 쌍의 데이터로 묶어줌
loader = DataLoader(dataset, batch_size=4, drop_last = False, shuffle=False, num_workers=0)
# dataset은 총 10개이고 batch_size = 4 이므로 각 batch크기는 4,4,2가 된다.
# drop_last가 False이므로 나머지 2개 데이터셋을 버리지 않고 batch로 만든다.  
sizes = []
for index, (xb, yb) in enumerate(loader, 1):
# batch번호는 index, 입력 데이터는 xb, 정답은 yb, batch번호는 1부터 시작.
# loader를 반복하면 3개의 batch가 나옴. 
    sizes.append([index, len(yb)])
    print(index, tuple(xb.shape), tuple(yb.shape))
print("합계:", sum(n for _, n in sizes))
# for 반복문 안의 _는 값을 받아오긴 하지만 사용하지는 않겠다는 프로그래머들 사이의 관례적인 변수명임.
# 필요한건 batch 사이즈이지 batch번호는 아니기 때문에 for문 안에  _ 을 사용함.
logits = torch.tensor([[2., 0.], [0., 2.]])
print("예시 CE:", round(nn.CrossEntropyLoss()(logits, torch.tensor([0, 1])).item(), 6))
#  nn.CrossEntropyLoss() => 해당 클래스의 객체를 생성
#  loss = nn.CrossEntropyLoss()
#  loss(logits, torch.tensor([0, 1])) = nn.CrossEntropyLoss()(logits, torch.tensor([0, 1]))
out = Path(__file__).resolve().parent / "outputs"
out.mkdir(exist_ok=True)
with (out / "batch_sizes.csv").open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f); writer.writerow(["batch", "samples"]); writer.writerows(sizes)

# 추가 A09: 배치 크기가 다를 때 평균 loss를 합치는 방법입니다.
# 입력: 평균 손실 [0.1, 0.2, 0.9], 각 배치의 표본 수 [4, 4, 2].
# 요구: loss × 실제 표본 수의 합을 전체 표본 수로 나누세요.
# 금지: 배치 평균 세 개를 단순 평균하거나 마지막 배치를 4건으로 계산하지 않습니다.
# 완료: weighted_loss=0.3. 잘못된 단순 평균 0.4와 차이를 설명합니다.
loss_means = [0.1, 0.2, 0.9]
sample_counts = [4, 4, 2]

weighted_loss = sum(loss*count for loss, count in zip(loss_means, sample_counts))
# 배치 크기를 고려한 가중 평균값 계산
print("샘플 가중 loss:", round(weighted_loss, 6))
print("배치 단순 평균:", round(sum(loss_means) / len(loss_means), 6))
