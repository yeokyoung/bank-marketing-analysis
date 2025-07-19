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

# 1. 전처리된 데이터 불러오기
# 참고: 전처리 코드를 이미 실행했다고 가정
# 이 스크립트는 전처리된 X_train_processed, X_test_processed, y_train, y_test가 있다고 가정합니다.

# 데이터 불러오기
print("데이터 로드 중...")
df = pd.read_csv("/Users/kim-yeokyoung/Desktop/세종대/4학년 1학기/기계학습/[기계학습] Datasets/Bank_Marketing/Bank_Marketing_Dataset.csv")
# 타겟 변수 인코딩
df['y_encoded'] = df['y'].map({'yes': 1, 'no': 0})

# 범주형 변수와 수치형 변수 분리
categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 
                       'contact', 'month', 'day_of_week', 'poutcome']
numeric_features = ['age', 'duration', 'campaign', 'pdays', 'previous', 
                   'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']

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

# 전처리 적용
print("전처리 적용 중...")
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"전처리 후 훈련 데이터 형태: {X_train_processed.shape}")
print(f"전처리 후 테스트 데이터 형태: {X_test_processed.shape}")

# 2. 모델 정의
print("\n모델 정의 중...")
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# 3. 모델 학습 및 평가
results = {}

for name, model in models.items():
    print(f"\n{name} 학습 중...")
    
    # 모델 학습
    model.fit(X_train_processed, y_train)
    
    # 예측
    y_pred = model.predict(X_test_processed)
    y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
    
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
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {name}')
    plt.legend(loc="lower right")
    plt.savefig(f'roc_curve_{name.replace(" ", "_")}.png')
    plt.close()

# 4. 모델 성능 비교
results_df = pd.DataFrame(results).T
print("\n모델 성능 비교:")
print(results_df)

# 성능 비교 시각화
plt.figure(figsize=(12, 8))
results_df.plot(kind='bar', figsize=(12, 8))
plt.title('모델 성능 비교')
plt.xlabel('모델')
plt.ylabel('점수')
plt.xticks(rotation=0)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)
plt.tight_layout()
plt.savefig('model_comparison.png')
plt.close()

# 5. 최고 성능 모델 선택
best_model_name = results_df['F1 Score'].idxmax()
print(f"\n최고 성능 모델 (F1 기준): {best_model_name}")

# 6. 하이퍼파라미터 튜닝 (최고 성능 모델)
print(f"\n{best_model_name} 하이퍼파라미터 튜닝 중...")

# 각 모델별 하이퍼파라미터 그리드 정의
param_grids = {
    'Logistic Regression': {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs']
    },
    'Decision Tree': {
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'Random Forest': {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    },
    'Gradient Boosting': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
}

# 최고 성능 모델의 하이퍼파라미터 그리드 선택
best_param_grid = param_grids[best_model_name]

# 그리드 서치 수행
grid_search = GridSearchCV(
    models[best_model_name], best_param_grid,
    cv=3, scoring='f1', n_jobs=-1, verbose=1
)

grid_search.fit(X_train_processed, y_train)

print(f"최적 하이퍼파라미터: {grid_search.best_params_}")
print(f"최고 F1 점수: {grid_search.best_score_:.4f}")

# 최적화된 모델로 테스트 세트 평가
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_processed)
y_pred_proba = best_model.predict_proba(X_test_processed)[:, 1]

# 평가 지표 계산
tuned_results = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1 Score': f1_score(y_test, y_pred),
    'ROC AUC': roc_auc_score(y_test, y_pred_proba)
}

print("\n튜닝된 모델 평가 결과:")
for metric, value in tuned_results.items():
    print(f"  {metric}: {value:.4f}")

# 혼동 행렬
cm = confusion_matrix(y_test, y_pred)
print("\n혼동 행렬:")
print(cm)

# 분류 보고서
print("\n분류 보고서:")
print(classification_report(y_test, y_pred))

# 7. 특성 중요도 시각화 (Random Forest 또는 Gradient Boosting)
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    # 특성 이름 가져오기
    try:
        cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
        feature_names = np.append(numeric_features, cat_features)
        
        # 특성 중요도 추출
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # 상위 20개 특성만 시각화
        top_k = min(20, len(feature_names))
        
        plt.figure(figsize=(12, 8))
        plt.title(f'{best_model_name} - 특성 중요도 (상위 {top_k}개)')
        plt.bar(range(top_k), importances[indices][:top_k])
        plt.xticks(range(top_k), feature_names[indices][:top_k], rotation=90)
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
        
        print("\n특성 중요도 (상위 10개):")
        for i in range(min(10, len(feature_names))):
            print(f"  {feature_names[indices][i]}: {importances[indices][i]:.4f}")
    except:
        print("\n특성 중요도를 계산할 수 없습니다.")

# 8. 최종 모델 저장 (선택 사항)
import joblib

print("\n최종 모델 및 전처리 파이프라인 저장 중...")
# 전체 파이프라인 저장
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', best_model)
])

joblib.dump(pipeline, 'bank_marketing_model_pipeline.pkl')
print("모델 파이프라인이 'bank_marketing_model_pipeline.pkl'로 저장되었습니다.")

print("\n모델링 완료!")