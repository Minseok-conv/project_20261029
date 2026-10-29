"""실습 03: 매번 새로운 그래프를 만들고 gradient를 비웁니다."""
from pathlib import Path
import csv
import torch

w = torch.tensor(0.0, requires_grad=True)
# requires_grad = True를 설정하면 해당 tensor에 대한 기울기를 계산하고 추적하도록 한다.
lr = 0.1
# lr = 학습률
history = []
# 각 스텝마다 갱신된 w값과 손실된 값을 저장하는 리스트
for step in range(1, 11):
    #학습을 10번 진행
    loss = (w - 4.0) ** 2
    loss.backward()
    with torch.no_grad():
        w -= lr * w.grad
# w를 이미 구한 gradient로 재설정하는 과정이므로 계산그래프에 포함 시킬 필요가 없다. 따라서 with torch.no_grad()구문에 포함시킴.
    w.grad.zero_()
# pytorch에서는 backward()를 여러번 수행하면 gradient가 기본적으로 누적됨.
# grad.zero()를 해주는 이유는 다음 backward() 수행 시 새로운 gradient값을 담기 위한 것임.

# step 1: gradient = -8.0
                #  ↓
            #  w 업데이트
                #  ↓
        #    gradient 초기화
                #  ↓
# step 2: gradient = -6.4
                #  ↓
             #  w 업데이트
                #  ↓
        #    gradient 초기화

    history.append([step, w.item(), (w.detach() - 4.0).square().item()])
for step, value, loss_after in history:
    print(step, round(value, 6), round(loss_after, 6))
out = Path(__file__).resolve().parent / "outputs"
out.mkdir(exist_ok=True)
with (out / "gd_history.csv").open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f); writer.writerow(["step", "w_after", "loss_after"]); writer.writerows(history)
