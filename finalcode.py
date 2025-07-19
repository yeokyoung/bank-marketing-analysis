import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# MacOS 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 1. 데이터 불러오기 - 경로 수정
df = pd.read_csv("/Users/kim-yeokyoung/Desktop/세종대/4학년 1학기/기계학습/[기계학습] Datasets/Bank_Marketing/Bank_Marketing_Dataset.csv")
print(f"데이터 크기: {df.shape}")

# 2. 데이터 탐색
# 기본 정보 확인
print("\n데이터 기본 정보:")
print(df.info())

# 기술 통계량
print("\n기술 통계량:")
print(df.describe())

# 결측치 확인
print("\n결측치 확인:")
print(df.isnull().sum())

# 타겟 변수 분포
print("\n타겟 변수 분포:")
print(df['y'].value_counts())
print(df['y'].value_counts(normalize=True).round(4) * 100)

# 3. 데이터 시각화
# 시각화 함수들 정의
def plot_categorical_distributions(df, cat_cols):
    """범주형 변수 분포 시각화"""
    n_cols = 2
    n_rows = (len(cat_cols) + 1) // 2
    
    plt.figure(figsize=(16, n_rows * 5))
    for i, col in enumerate(cat_cols, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.countplot(x=col, data=df, hue='y')
        plt.title(f'{col} 분포', fontsize=14)
        plt.xticks(rotation=45)
        plt.legend(title='예금 가입')
    
    plt.tight_layout()
    plt.savefig('categorical_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_numeric_distributions(df, num_cols):
    """수치형 변수 분포 시각화"""
    n_cols = 2
    n_rows = (len(num_cols) + 1) // 2
    
    plt.figure(figsize=(16, n_rows * 5))
    for i, col in enumerate(num_cols, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.histplot(data=df, x=col, hue='y', kde=True, element='step')
        plt.title(f'{col} 분포', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('numeric_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_correlation_matrix(df, num_cols):
    """상관관계 행렬 시각화"""
    plt.figure(figsize=(14, 12))
    corr = df[num_cols + ['y_encoded']].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                vmin=-1, vmax=1, center=0, square=True, linewidths=.5)
    plt.title('수치형 변수 상관관계', fontsize=18)
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

# 범주형 변수와 수치형 변수 분리
categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 
                       'contact', 'month', 'day_of_week', 'poutcome']
numeric_features = ['age', 'duration', 'campaign', 'pdays', 'previous', 
                   'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']

# 타겟 변수 인코딩 (시각화 용도)
df['y_encoded'] = df['y'].map({'yes': 1, 'no': 0})

# 시각화 실행
plot_categorical_distributions(df, categorical_features)
plot_numeric_distributions(df, numeric_features)
plot_correlation_matrix(df, numeric_features)

# 4. 데이터 전처리
# 특성과 타겟 분리
X = df.drop(['y', 'y_encoded'], axis=1)
y = df['y_encoded']

# 훈련/테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 전처리 파이프라인 구성
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 5. 모델 정의
models = {
    'Logistic Regression': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
    ]),
    
    'Decision Tree': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(random_state=42, class_weight='balanced'))
    ]),
    
    'Random Forest': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
    ]),
    
    'Gradient Boosting': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(random_state=42))
    ])
}

# 6. 모델 학습 및 평가
results = {}

for name, model in models.items():
    print(f"\n{name} 학습 중...")
    
    # 모델 학습
    model.fit(X_train, y_train)
    
    # 예측
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 평가 지표 계산
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # 결과 저장
    results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc
    }
    
    print(f"{name} 평가 결과:")
    print(f"  정확도: {accuracy:.4f}")
    print(f"  정밀도: {precision:.4f}")
    print(f"  재현율: {recall:.4f}")
    print(f"  F1 점수: {f1:.4f}")
    print(f"  ROC AUC: {roc_auc:.4f}")
    
    # 혼동 행렬
    cm = confusion_matrix(y_test, y_pred)
    print("\n  혼동 행렬:")
    print(cm)
    
    # 분류 보고서
    print("\n  분류 보고서:")
    print(classification_report(y_test, y_pred))
    
    # ROC 곡선 그리기
    plt.figure(figsize=(10, 8))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {name}', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.savefig(f'roc_curve_{name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.close()

# 7. 모델 성능 비교
results_df = pd.DataFrame(results).T
print("\n모델 성능 비교:")
print(results_df)

# 성능 비교 시각화
plt.figure(figsize=(14, 10))
results_df.plot(kind='bar', figsize=(14, 10))
plt.title('모델 성능 비교', fontsize=18)
plt.xlabel('모델', fontsize=14)
plt.ylabel('점수', fontsize=14)
plt.xticks(rotation=0, fontsize=12)
plt.yticks(fontsize=12)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=5, fontsize=12)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 8. 최고 성능 모델 선택
best_model_name = results_df['F1 Score'].idxmax()
print(f"\n최고 성능 모델: {best_model_name}")

# 9. 하이퍼파라미터 튜닝 (Random Forest 기준)
if best_model_name == 'Random Forest':
    print("\nRandom Forest 하이퍼파라미터 튜닝 중...")
    
    rf_pipeline = models['Random Forest']
    
    # 그리드 서치 파라미터
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2]
    }
    
    grid_search = GridSearchCV(
        rf_pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"최적 하이퍼파라미터: {grid_search.best_params_}")
    print(f"최고 F1 점수: {grid_search.best_score_:.4f}")
    
    # 최적화된 모델로 테스트 세트 평가
    best_rf = grid_search.best_estimator_
    y_pred = best_rf.predict(X_test)
    y_pred_proba = best_rf.predict_proba(X_test)[:, 1]
    
    print("\n최적화된 Random Forest 평가 결과:")
    print(f"  정확도: {accuracy_score(y_test, y_pred):.4f}")
    print(f"  정밀도: {precision_score(y_test, y_pred):.4f}")
    print(f"  재현율: {recall_score(y_test, y_pred):.4f}")
    print(f"  F1 점수: {f1_score(y_test, y_pred):.4f}")
    print(f"  ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    # 특성 중요도 시각화
    rf_classifier = best_rf.named_steps['classifier']
    
    # 전처리된 특성 이름 가져오기
    preprocessor = best_rf.named_steps['preprocessor']
    cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
    feature_names = np.append(numeric_features, cat_features)
    
    # 특성 중요도 시각화
    importances = rf_classifier.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(14, 10))
    plt.title('Random Forest - 특성 중요도', fontsize=18)
    plt.bar(range(len(indices[:20])), importances[indices[:20]])
    plt.xticks(range(len(indices[:20])), feature_names[indices[:20]], rotation=90, fontsize=12)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

# 10. 비즈니스 인사이트 도출
print("\n비즈니스 인사이트:")
print("1. 마케팅 캠페인 효율성 향상을 위한 주요 특성:")
print("   - 통화 지속 시간이 길수록 예금 가입 가능성이 높아짐")
print("   - 이전 캠페인 결과가 성공적이었던 고객은 다시 가입할 가능성이 높음")
print("   - 특정 직업군과 교육 수준을 가진 고객들의 가입률이 높음")
print("2. 타겟 마케팅 전략:")
print("   - 고가입 가능성이 높은 고객 세그먼트에 집중")
print("   - 통화 품질과 지속 시간 확보를 위한 상담원 교육")
print("3. 마케팅 캠페인 최적화:")
print("   - 계절적 요인과 경제 지표를 고려한 캠페인 타이밍 조정")
print("   - 연락 방법과 빈도 최적화")

# 11. 결론 및 제한점
print("\n결론 및 제한점:")
print("1. 모델 성능:")
print(f"   - 최고 성능 모델: {best_model_name}")
print(f"   - F1 점수: {results[best_model_name]['F1 Score']:.4f}")
print("2. 제한점:")
print("   - 클래스 불균형 문제 (예금 미가입 데이터가 가입 데이터보다 많음)")
print("   - 시간에 따른 고객 행동 변화 반영 어려움")
print("3. 개선 방향:")
print("   - 추가 특성 공학을 통한 모델 성능 향상")
print("   - 앙상블 기법 적용 및 최적화")
print("   - 더 많은 데이터 수집을 통한 모델 일반화 능력 향상")