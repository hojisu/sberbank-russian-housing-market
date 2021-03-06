# Sberbank Russian Housing Market  

집 값에 영향을 끼치는 요소들을 파악하고자 EDA 및 데이터 분석을 통한 선형/비선형 회귀 모형 개발 

https://www.kaggle.com/c/sberbank-russian-housing-market  

## Contents  
### Analysis Description  

### [01. Exploratory Data Analysis](https://github.com/hojisu/sberbank-russian-housing-market/blob/master/01-Exploratory-Data-Analysis/01-Exploratory-Data-Analysis.ipynb)  
### [02. Preprocessing & Feature Engineering](https://github.com/hojisu/sberbank-russian-housing-market/blob/master/02-Preprocessing-Feature-Engineering/02-Preprocessing-Feature-Engineering.ipynb)  
### [03. Modeling with StatsModels](https://github.com/hojisu/sberbank-russian-housing-market/blob/master/03-Modeling-with-StatsModels/03-Modeling-with-StatsModels.ipynb)  
### [04. Modeling with XGBoost](https://github.com/hojisu/sberbank-russian-housing-market/blob/master/04-Modeling-with-XGBoost/04_Modeling-with-XGBoost.ipynb)  
### [05. Sberbank Russian Housing Market Price Prediction PPT](https://github.com/hojisu/sberbank-russian-housing-market/blob/master/05-Sberbank-Russian-Housing-Market-Price-Prediction-PPT/Sberbank-Russian-Housing-Market-Price-Prediction-PPT.pdf)

***

## 데이터
캐글에서 제공되는 Sberbank의 러시아 모스코 주택 시장의 가격 변동 예측 문제를 선택하였습니다. 예측 가격과 실제 가격 간의 에러를 RMSLE(Root Mean Squared Logarithmic Error)로 측정 하였습니다. 인풋 데이터로는 1인당 소득, 외환 환율, GDP 등을 포함한 시계열 거시경제 데이터와 평수, 층수, 주변 환경 등 300가지 집 특성을 가진 3만 건의 주택거래 내역이 주어졌습니다.

## Feature Engineering
데이터 타입은 정수, 실수, 카테고리값으로 이루어져 있습니다. 카테고리 변수들은 Yes/No 아니면 순서를 가진 데이터로 Label Encoding을 적용하여 정수값으로 변환하였습니다. 비어있는 데이터가 많이 존재한다는 것을 파악하고 상관관계가 높은 변수들끼리 선형회귀 하여 채웠습니다. 덜 채워진 수치 데이터는 평균값으로 카테고리 데이터는 모드값으로 대신 하였습니다. 또한, 각각의 독립변수들의 분포를 살펴보고 skewness가 1 이상인 것과 이분산성이 나타나는 변수들은 로그를 적용하여 정규분포에 가깝게 만들어 주었습니다. 선형 회귀 시 조건수가 크게 나와 스케일링이 필요 하다고 판단하였고 독립변수 간에 상관관계를 확인해 볼 필요가 있었습니다.

## 차원축소
독립변수가 약 350개에 달해 차원축소가 필요하다고 판단하였습니다. F 검정을 통해 각 독립변수가 종속변수에 가진 영향력을 살펴보고 p-value가 0.05보다 크면 중요도가 작다고 판단하여 제거하였습니다. 또한 다중공선성을 없애기 위해 Variance Inflation Factor를 계산하여 Greedy 방법으로 독립변수들을 하나씩 줄여 차원축소를 진행하였습니다.

## Outlier와 잔차 정규성 테스트
회귀 성능 향상을 위해서 가격예측에 영향을 주는 큰 레버리지를 가진 데이터를 찾아야 했습니다. Cook’s Distance를 사용하여 회귀 분석 시 잔차와 레버리지가 큰 데이터들을 살펴보았습니다.QQ Plot과 Residual Plot을 그려서 잔차가 정규성을 따르는지 확인하고 부분회귀 플롯 및 교차 검증을 통해 성능을 확인하였습니다.

## 모델링
기본적인 전처리 후 StatsModels의 Ordinary Least Square와 정규화 적용된 Ordinary Least Square, XGBoost 모델들을 사용하여 퍼포먼스를 비교하였습니다. 비선형 모델인 XGBoost의 feature importance를 통해 회귀분석에서 선택된 요소들의 차이점도 확인하였습니다.

## 개선 할 점
OLS와 XGBoost 모델의 퍼포먼스의 차이가 근소하였습니다. 반복적인 Feature Engineering과 모델 튜닝이 필요합니다. 

