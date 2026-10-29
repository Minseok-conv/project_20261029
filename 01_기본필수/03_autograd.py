"""실습 03: x=3에서 y=x²+2x+1과 기울기를 계산합니다."""
import torch
x = torch.tensor(3.0, requires_grad=True)
y = x**2 + 2*x + 1
y.backward()
print("y:", y.item())
print("dy/dx:", x.grad.item())
