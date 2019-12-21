# Sberbank Russian Housing Market  

Predict realty price fluctuations in Russia’s volatile economy  

https://www.kaggle.com/c/sberbank-russian-housing-market  

## Contents  
### Analysis Description  
- 한국어  

### [01. Exploratory Data Analysis](https://github.com/hojisu/sberbank-russian-housing-market/tree/master/01-Exploratory-Data-Analysis)  
### [02. Preprocessing & Feature Engineering](https://github.com/hojisu/sberbank-russian-housing-market/tree/master/02-Preprocessing-Feature-Engineering)  
### [03. Modeling with StatsModels](https://github.com/hojisu/sberbank-russian-housing-market/tree/master/03-Modeling-StatsModels)  
### [04. Modeling with XGBoost(Scikit-learn Wrapper)](https://github.com/hojisu/sberbank-russian-housing-market/tree/master/04-Modeling-with-Scikit-Learn-Regressor)  

***

## 데이터
회귀 프로젝트로 캐글에서 제공되는 Sberbank의 러시아 모스코 주택 시장의 가격 변동 예측 문제를 선택하였습니다. 예측 가격과 실제 가격 간의 에러를 RMSLE(Root Mean Squared Logarithmic Error)로 측정 하였습니다. 인풋 데이터로는 1인당 소득, 외환 환율, GDP 등을 포함한 시계열 거시경제 데이터와 평수, 층수, 주변 환경 등 300가지 집 특성을 가진 3만 건의 주택거래 내역이 주어졌습니다.

## Feature Engineering
데이터 타입은 정수, 실수, 카테고리값으로 이루어져 있습니다. 카테고리 변수들은 Yes/No 아니면 순서를 가진 데이터로 Label Encoding을 적용하여 정수값으로 변환하였습니다. 비어있는 데이터가 많이 존재한다는 것을 파악하고 상관관계가 높은 변수들끼리 선형회귀 하여 채웠습니다. 덜 채워진 수치 데이터는 평균값으로 카테고리 데이터는 모드값으로 대신 하였습니다. 또한, 각각의 독립변수들의 분포를 살펴보고 skewness가 1 이상인 것과 이분산성이 나타나는 변수들은 로그를 적용하여 정규분포에 가깝게 만들어 주었습니다. 선형 회귀 시 조건수가 크게 나와 스케일링이 필요 하다고 판단하였고 독립변수 간에 상관관계를 확인해 볼 필요가 있었습니다.

## 차원축소
독립변수가 약 350개에 달해 차원축소가 필요하다고 판단하였습니다. F 검정을 통해 각 독립변수가 종속변수에 가진 영향력을 살펴보고 p-value가 0.05보다 크면 중요도가 작다고 판단하여 제거하였습니다. 또한 다중공선성을 없애기 위해 Variance Inflation Factor를 계산하여 Greedy 방법으로 독립변수들을 하나씩 줄여 차원축소를 진행하였습니다.

## Outlier와 잔차 정규성 테스트
회귀 성능 향상을 위해서 가격예측에 영향을 주는 큰 레버리지를 가진 데이터를 찾아야 했습니다. Cook’s Distance를 사용하여 회귀 분석 시 잔차와 레버리지가 큰 데이터들을 살펴보았습니다. 교차 검증을 통해 주택 가격이 1, 2 백만 루블인 데이터들이 잔차가 크고 레버리지가 높은 것을 알 수 있었습니다. QQ Plot 그리고 Residual Plot을 그려 보니 이 두 가지 가격에서 뚜렷한 큰 오류가 발생하는 것으로 나타났습니다.

## 모델링
초기 접근 방식은 기본적인 전처리 후 StatsModels의 Ordinary Least Square부터 Scikit-Learn의 Elastic Nets, DecisionTree Regressor, RandomForest Regressor, Support Vector Regressor, XGBoost를 사용하여 퍼포먼스가 가장 좋은 모델을 선택하고 반복적인 피쳐 엔지니어링과 모델 튜닝을 진행하였습니다. 모델별 10겹 cross validation을 통해 XGBoost의 RMSLE가 가장 낮게 나온 것을 확인하였습니다.

## 개선 할 점
XGBoost의 Cross validation한 결과와 캐글에 submission한 score값의 차이가 컸습니다. XGBoost에서의 과최적화가 발생한 것으로 보입니다. XGBoost의 과최적화 방지에 관하여 좀 더 연구할 필요가 있습니다. 

