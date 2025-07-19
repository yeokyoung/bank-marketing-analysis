import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# MacOS 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 1. 데이터 불러오기
print("데이터 로드 중...")
# 수정된 파일 경로 - Bank_Marketing 폴더만 지정, CSV 파일 이름 추가
df = pd.read_csv("/Users/kim-yeokyoung/Desktop/세종대/4학년 1학기/기계학습/[기계학습] Datasets/Bank_Marketing/Bank_Marketing_Dataset.csv")
print(f"데이터 크기: {df.shape}")

# 2. 데이터 기본 탐색
# 기본 정보 확인
print("\n데이터 기본 정보:")
print(df.info())

# 처음 몇 행 확인
print("\n데이터 샘플:")
print(df.head())

# 기술 통계량
print("\n기술 통계량:")
print(df.describe())

# 결측치 확인
print("\n결측치 확인:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0] if missing_values.any() > 0 else "결측치 없음")

# 타겟 변수 분포
print("\n타겟 변수 분포:")
print(df['y'].value_counts())
print(df['y'].value_counts(normalize=True).round(4) * 100)

# 3. 데이터 시각화
# 시각화를 위한 설정
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# 3.1 범주형 변수 시각화
categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 
                       'contact', 'month', 'day_of_week', 'poutcome']

plt.figure(figsize=(10, 8))
sns.countplot(y='job', data=df, hue='y', order=df['job'].value_counts().index)
plt.title('직업별 예금 가입 분포')
plt.tight_layout()
plt.savefig('job_distribution.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.countplot(x='marital', data=df, hue='y')
plt.title('결혼 상태별 예금 가입 분포')
plt.tight_layout()
plt.savefig('marital_distribution.png')
plt.close()

# 3.2 수치형 변수 시각화
numeric_features = ['age', 'duration', 'campaign', 'pdays', 'previous', 
                   'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']

plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='age', hue='y', kde=True, element='step')
plt.title('나이별 예금 가입 분포')
plt.tight_layout()
plt.savefig('age_distribution.png')
plt.close()

plt.figure(figsize=(12, 6))
sns.boxplot(x='y', y='duration', data=df)
plt.title('예금 가입 여부에 따른 통화 지속시간 분포')
plt.tight_layout()
plt.savefig('duration_distribution.png')
plt.close()

# 3.3 상관관계 분석
# 타겟 변수 인코딩 (분석용)
df['y_encoded'] = df['y'].map({'yes': 1, 'no': 0})

# 수치형 변수 상관관계
plt.figure(figsize=(12, 10))
numeric_df = df[numeric_features + ['y_encoded']]
corr = numeric_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
           vmin=-1, vmax=1, center=0, square=True, linewidths=.5)
plt.title('수치형 변수 상관관계')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

# 4. 데이터 전처리
print("\n데이터 전처리 시작...")

# 4.1 특성과 타겟 분리
X = df.drop(['y', 'y_encoded'], axis=1)
y = df['y_encoded']

# 4.2 데이터 분할 (훈련/테스트)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"훈련 데이터 크기: {X_train.shape}")
print(f"테스트 데이터 크기: {X_test.shape}")

# 4.3 전처리 파이프라인 구성
# 수치형 변수 전처리
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 범주형 변수 전처리
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# 전처리 파이프라인 결합
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 4.4 전처리 적용
print("전처리 파이프라인 적용 중...")
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"전처리 후 훈련 데이터 형태: {X_train_processed.shape}")
print(f"전처리 후 테스트 데이터 형태: {X_test_processed.shape}")

# 4.5 전처리된 데이터 확인 (샘플)
# 특성 이름 가져오기 (인코딩 후)
try:
    cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
    feature_names = np.append(numeric_features, cat_features)
    print(f"\n전처리 후 특성 수: {len(feature_names)}")
    print(f"처음 10개 특성 이름: {feature_names[:10]}")
except:
    print("\n전처리 후 특성 이름을 가져올 수 없습니다.")

print("\n데이터 전처리 완료!")

# 5. 전처리 과정 요약
print("\n전처리 과정 요약:")
print("1. 결측치 처리: 수치형 변수는 중앙값, 범주형 변수는 최빈값으로 대체")
print("2. 수치형 변수 스케일링: 표준화(StandardScaler) 적용")
print("3. 범주형 변수 인코딩: 원-핫 인코딩(OneHotEncoder) 적용")
print("4. 데이터 분할: 80% 훈련, 20% 테스트")
print(f"5. 최종 특성 수: {X_train_processed.shape[1]}")