## 1. Dataset
데이터는 UCI에서 참조하였다. 
<br/>[The Human Activity Recognition 70+ (HAR70+)](https://archive.ics.uci.edu/dataset/780/har70)
<br/><br/>
- Purpose: HAR(Human Activity Recognition)를 위해 설계된 머신 러닝 모델을 학습하기 위해 가속도계 데이터를 수집함. 모델은 건강한 노인부터 허약한 노인까지의 활동을 인식하기 위하여 고안되었다.
- Data:  18명의 노인 참가자가 두 개(등,허벅지)의 3축 가속도계를 착용하고 자유 생활 환경에서 기록
- Sampling rate: 50Hz
- Features: 총 6개의 feature vector를 가진다. 가속도의 단위는 \[g\]이다.
  1. timestamp: 기록된 샘플의 날짜 및 시간
  2. back_x: x방향(아래)에서의 허리 센서 가속도
  3. back_y: y방향(왼쪽)에서의 허리 센서 가속도
  4. back_z: z방향(앞쪽)에서의 허리 센서 가속도
  5. thigh_x: x방향(아래)에서의 허벅지 센서 가속도
  6. thigh_y: y방향(오른쪽)에서의 허벅지 센서 가속도
  7. thigh_z: z방향(뒤쪽)에서의 허벅지 센서 가속도
- Labels: 각 activity를 나타낸다.<br/>
  1: 걷기<br/>
  3: 발을 질질 끌기<br/>
  4: 계단 오르기<br/>
  5: 계단 내리기<br/>
  6: 서있기<br/>
  7: 앉기<br/>
  8: 눕기<br/>

## 2. 전처리
Feature vector의 개수는 각 참가자마다 대략 10만 개이다. 
총 18명의 참가자의 데이터를 학습해야하는데, 데이터의 양이 방대해져 시간이 오래 걸리므로 50Hz(0.02s) 단위로 수집된 데이터를 초 단위로 feature vector에 대해 평균을 내기로 하였다.
이 때 label의 평균을 낼 때에는 **One hot encoding probability** 방식을 사용하였다. 이는 각 class가 특정 확률을 가지도록 표현된 vector이며, 각 label이 초 단위로 어느 정도의 확률을 가지고 등장하는지를 나타낸다. 
다음은 이 방식의 에시이다.
<br/>
<center>
  <img src="https://github.com/user-attachments/assets/d1c6d34a-7793-43f6-b161-189597f8650f" alt="image" width="400"/>
</center>
<br/><br/>
각 참가자마다 대략 2600개의 초에 대해 평균을 내었다. 본 연구의 모델은 정해진 시간 간격 S 동안 인간의 활동을 학습하고, 마지막 시간에 인간이 무슨 활동을 하고 있는지를 지도 학습을 통해 학습한다. 
이를 위해 S = 30sec으로 두고 5초의 stride을 두어 대략 6000개의 feature vector를 최종적으로 얻었다. 다음은 데이터 전처리에 대한 도식도이다.
<br/><br/>
<img src="https://github.com/user-attachments/assets/3d33afc9-5839-422e-9c00-8bd0043ba260" alt="image" width="800"/>

## 3. 모델 구현
모델 구현에는 WaveNet을 적용하였다. WaveNet에 대해서는 *URP/Documents*에서 찾아볼 수 있다. 데이터셋에서 label의 확률 분포를 encode하였으므로, WaveNet을 사용하기 적합하다. 
기존 WaveNet과의 차이점은, WaveNet은 input과 output의 size가 같지만 본 모델은 S 간격(input size)으로 들어온 데이터에 대해 마지막 시간만의 label을 결과로 낸다는 것이다. 
<br/><br/>
WaveNet의 효율을 분석하기 위해, 다른 간단한 구조인 MLP(Multi-Layer Perceptron)을 같이 구현하여 비교해보았다. 
MLP는 WaveNet과 달리 시간의 인과성을 고려하지 못하며, autoregressive하지 않아 이전의 데이터를 활용하지 못한다. 
MLP의 hidden layer의 개수는 2개이며, input과 output 를 포함하여 모든 layer에는 선형 변환을 사용하였다. 다음은 MLP의 구조를 도식화한 그림이다.<br/>
<img src="https://github.com/user-attachments/assets/0bf4cf5c-a9ca-4d68-8e62-38aaebed5512" alt="image" width="800"/>
<br><br>
WaveNet에서는 fc(fully-connected) layer이 input과 output layer로 사용되었으며,
hidden layer에는 convolutional layer가 6개 사용되었다. 이때 dilation factor를 [1, 2, 4, 8, 16, 32]로 설정함으로써 dilated convolutional layer를 만들었다. 또한 앞 시간만을 참조하기 위해 shifting을 사용하였고, 그만큼 0을 padding하였다.
Loss function에는 KL divergence을 사용하였다.

## 4. 결과 분석
<img src="https://github.com/user-attachments/assets/2a4f802f-bdd0-4f3d-8d1e-0844ba2ee4e9" alt="image" width="700"/>
<br/><br>
학습 결과, MLP는 약 88%에, WaveNet은 91%에 수렴하는 효율성을 보여주었다. WaveNet에서의 효율성이 더 높은 이유 중 하나로 시간의 인과성 이용을 들어볼 수 있다. 
MLP는 각 timestamp에 대해 독립적으로, WaveNet은 시간의 상호관계를 고려하여 데이터를 처리하기 때문에 이와 같은 차이가 나타난다고 볼 수 있다.
