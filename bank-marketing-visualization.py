import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# MacOS 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 데이터 로드
print("데이터 로드 중...")
df = pd.read_csv("/Users/kim-yeokyoung/Desktop/세종대/4학년 1학기/기계학습/기계학습 과제/[기계학습] Datasets/Bank_Marketing/Bank_Marketing_Dataset.csv")

# 타겟 변수 인코딩 (시각화 용도)
df['y_encoded'] = df['y'].map({'yes': 1, 'no': 0})

# 범주형 변수와 수치형 변수 정의
categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 
                       'contact', 'month', 'day_of_week', 'poutcome']
numeric_features = ['age', 'duration', 'campaign', 'pdays', 'previous', 
                   'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']

# 시각화를 위한 설정
plt.style.use('ggplot')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

print("데이터 시각화 시작...")

# 1. 타겟 변수 분포 시각화
plt.figure(figsize=(10, 6))
target_counts = df['y'].value_counts()
plt.pie(target_counts, labels=target_counts.index, autopct='%1.1f%%', startangle=90, 
        colors=['#ff9999','#66b3ff'])
plt.title('예금 가입 여부 분포')
plt.axis('equal')
plt.savefig('target_distribution.png')
plt.close()

# 2. 범주형 변수 시각화
# 2.1 직업별 분포
plt.figure(figsize=(14, 8))
job_order = df['job'].value_counts().index
sns.countplot(y='job', data=df, hue='y', order=job_order)
plt.title('직업별 예금 가입 분포')
plt.xlabel('고객 수')
plt.ylabel('직업')
plt.legend(title='예금 가입', loc='lower right')
plt.tight_layout()
plt.savefig('job_distribution.png')
plt.close()

# 2.2 교육 수준별 분포
plt.figure(figsize=(14, 8))
edu_order = df['education'].value_counts().index
sns.countplot(y='education', data=df, hue='y', order=edu_order)
plt.title('교육 수준별 예금 가입 분포')
plt.xlabel('고객 수')
plt.ylabel('교육 수준')
plt.legend(title='예금 가입', loc='lower right')
plt.tight_layout()
plt.savefig('education_distribution.png')
plt.close()

# 2.3 연락 방법별 분포
plt.figure(figsize=(10, 6))
contact_counts = df.groupby(['contact', 'y']).size().unstack()
contact_counts.plot(kind='bar', stacked=False)
plt.title('연락 방법별 예금 가입 분포')
plt.xlabel('연락 방법')
plt.ylabel('고객 수')
plt.xticks(rotation=0)
plt.legend(title='예금 가입')
plt.tight_layout()
plt.savefig('contact_distribution.png')
plt.close()

# 2.4 이전 캠페인 결과별 분포
plt.figure(figsize=(10, 6))
poutcome_counts = df.groupby(['poutcome', 'y']).size().unstack()
poutcome_counts.plot(kind='bar', stacked=False)
plt.title('이전 캠페인 결과별 예금 가입 분포')
plt.xlabel('이전 캠페인 결과')
plt.ylabel('고객 수')
plt.xticks(rotation=0)
plt.legend(title='예금 가입')
plt.tight_layout()
plt.savefig('poutcome_distribution.png')
plt.close()

# 3. 수치형 변수 시각화
# 3.1 나이 분포
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='age', hue='y', kde=True, element='step', bins=30)
plt.title('나이별 예금 가입 분포')
plt.xlabel('나이')
plt.ylabel('고객 수')
plt.legend(title='예금 가입')
plt.tight_layout()
plt.savefig('age_distribution.png')
plt.close()

# 3.2 통화 지속시간 분포
plt.figure(figsize=(12, 6))
sns.boxplot(x='y', y='duration', data=df)
plt.title('예금 가입 여부에 따른 통화 지속시간 분포')
plt.xlabel('예금 가입')
plt.ylabel('통화 지속시간 (초)')
plt.tight_layout()
plt.savefig('duration_boxplot.png')
plt.close()

plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='duration', hue='y', kde=True, element='step', bins=30)
plt.title('통화 지속시간별 예금 가입 분포')
plt.xlabel('통화 지속시간 (초)')
plt.ylabel('고객 수')
plt.legend(title='예금 가입')
plt.tight_layout()
plt.savefig('duration_distribution.png')
plt.close()

# 3.3 캠페인 연락 횟수 분포
plt.figure(figsize=(12, 6))
sns.boxplot(x='y', y='campaign', data=df)
plt.title('예금 가입 여부에 따른 캠페인 연락 횟수 분포')
plt.xlabel('예금 가입')
plt.ylabel('연락 횟수')
plt.tight_layout()
plt.savefig('campaign_boxplot.png')
plt.close()

# 4. 상관관계 분석
# 수치형 변수 상관관계
plt.figure(figsize=(14, 12))
numeric_df = df[numeric_features + ['y_encoded']]
corr = numeric_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
           vmin=-1, vmax=1, center=0, square=True, linewidths=.5)
plt.title('수치형 변수 상관관계')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

# 5. 목표 변수와의 관계 시각화
# 5.1 월별 가입률
plt.figure(figsize=(14, 7))
month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
monthly_signup_rate = df.groupby('month')['y_encoded'].mean() * 100

# month_order에 있는 월 중에서 df['month']에 있는 월만 필터링
available_months = [m for m in month_order if m in df['month'].unique()]
monthly_signup_rate = monthly_signup_rate.reindex(available_months)

ax = monthly_signup_rate.plot(kind='bar', color='skyblue')
plt.title('월별 예금 가입률')
plt.xlabel('월')
plt.ylabel('가입률 (%)')
plt.xticks(rotation=45)

# 가입률 숫자 표시
for i, v in enumerate(monthly_signup_rate):
    ax.text(i, v + 0.5, f'{v:.1f}%', ha='center')

plt.tight_layout()
plt.savefig('monthly_signup_rate.png')
plt.close()

# 5.2 요일별 가입률
plt.figure(figsize=(10, 6))
day_order = ['mon', 'tue', 'wed', 'thu', 'fri']
daily_signup_rate = df.groupby('day_of_week')['y_encoded'].mean() * 100

# day_order에 있는 요일 중에서 df['day_of_week']에 있는 요일만 필터링
available_days = [d for d in day_order if d in df['day_of_week'].unique()]
daily_signup_rate = daily_signup_rate.reindex(available_days)

ax = daily_signup_rate.plot(kind='bar', color='lightgreen')
plt.title('요일별 예금 가입률')
plt.xlabel('요일')
plt.ylabel('가입률 (%)')
plt.xticks(rotation=0)

# 가입률 숫자 표시
for i, v in enumerate(daily_signup_rate):
    ax.text(i, v + 0.3, f'{v:.1f}%', ha='center')

plt.tight_layout()
plt.savefig('daily_signup_rate.png')
plt.close()

# 5.3 직업별 가입률
plt.figure(figsize=(14, 8))
job_signup_rate = df.groupby('job')['y_encoded'].mean() * 100
job_signup_rate = job_signup_rate.sort_values(ascending=False)

ax = job_signup_rate.plot(kind='bar', color='coral')
plt.title('직업별 예금 가입률')
plt.xlabel('직업')
plt.ylabel('가입률 (%)')
plt.xticks(rotation=45)

# 가입률 숫자 표시
for i, v in enumerate(job_signup_rate):
    ax.text(i, v + 0.5, f'{v:.1f}%', ha='center')

plt.tight_layout()
plt.savefig('job_signup_rate.png')
plt.close()

# 5.4 교육 수준별 가입률
plt.figure(figsize=(14, 8))
edu_signup_rate = df.groupby('education')['y_encoded'].mean() * 100
edu_signup_rate = edu_signup_rate.sort_values(ascending=False)

ax = edu_signup_rate.plot(kind='bar', color='mediumpurple')
plt.title('교육 수준별 예금 가입률')
plt.xlabel('교육 수준')
plt.ylabel('가입률 (%)')
plt.xticks(rotation=45)

# 가입률 숫자 표시
for i, v in enumerate(edu_signup_rate):
    ax.text(i, v + 0.5, f'{v:.1f}%', ha='center')

plt.tight_layout()
plt.savefig('education_signup_rate.png')
plt.close()

# 6. 경제 지표와 가입률의 관계
plt.figure(figsize=(12, 8))
plt.scatter(df['euribor3m'], df['cons.conf.idx'], c=df['y_encoded'], 
           alpha=0.5, cmap='coolwarm', s=30)
plt.colorbar(label='예금 가입 여부')
plt.title('유로 금리와 소비자 신뢰 지수에 따른 예금 가입 여부')
plt.xlabel('유로 3개월 금리')
plt.ylabel('소비자 신뢰 지수')
plt.tight_layout()
plt.savefig('economic_indicators.png')
plt.close()

print("데이터 시각화 완료!")
print("생성된 시각화 파일:")
print("- target_distribution.png: 타겟 변수 분포")
print("- job_distribution.png: 직업별 예금 가입 분포")
print("- education_distribution.png: 교육 수준별 예금 가입 분포")
print("- contact_distribution.png: 연락 방법별 예금 가입 분포")
print("- poutcome_distribution.png: 이전 캠페인 결과별 예금 가입 분포")
print("- age_distribution.png: 나이별 예금 가입 분포")
print("- duration_boxplot.png: 통화 지속시간 상자 그림")
print("- duration_distribution.png: 통화 지속시간별 예금 가입 분포")
print("- campaign_boxplot.png: 캠페인 연락 횟수 상자 그림")
print("- correlation_matrix.png: 상관관계 행렬")
print("- monthly_signup_rate.png: 월별 예금 가입률")
print("- daily_signup_rate.png: 요일별 예금 가입률")
print("- job_signup_rate.png: 직업별 예금 가입률")
print("- education_signup_rate.png: 교육 수준별 예금 가입률")
print("- economic_indicators.png: 경제 지표와 가입률의 관계")
