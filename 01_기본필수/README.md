
1. `.backward()`와 `.step()`은 각각 무엇을 바꾸나요? → 
backward()는 loss로부터 역전파(backpropagation)를 수행해서 각 학습 파라미터의 gradient를 계산하는 함수이다.
 step()은 계산된 gradient를 이용해서 모델의 파라미터(weight, bias)를 실제로 업데이트하는 것
2. `zero_grad()`를 생략하면 어떤 일이 생기나요? →
이전 step에  계산해서 저장된 미분 값에 누적된다. 
3. `train 정확도`와 `valid 정확도`는 어떤 자료를 사용하나요? →
train 정확도는 모델이 학습에 사용한 train 데이터를 사용한다. train 정확도는 모델이 학습한 문제를 얼마나 잘 맞히는지를 알 수 있다. 
valid 정확도는 모델이 학습에 사용하지 않은 valid 데이터를 사용한다. 이는 새로운 데이터에도 잘 작동하는지, 모델의 일반화 정도를 알 수 있다. 
4. `eval()`만으로 gradient 기록이 꺼지나요? → 
아니다. gradient기록을 더이상 진행하지 않으려면 with torch.no_grad(): 안에 계산그래프를 생성하지 않은 구문을 적어야 한다. 
5. 정상 데이터 4건에서 정확도1.0이면 실제 서비스도 잘 동작한다고 할 수 있나요? →
아니다. 우선 데이터 4건은 평가를 위한 개수로는 너무 적다. 또한 모델이 train 데이터에 과적합 되어 test 데이터에서 무조건 정답으로 출력할 수 있다. 따라서 모델이 실제 서비스에도 잘 동작하는지 판단하려면, test data에서 precision만 보는 것이 아니라 recall, f1-score, confusion matrix등도 함께 고려해야 한다. 



