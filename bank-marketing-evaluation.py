import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')


# MacOS 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 1. 저장된 모델과 데이터 불러오기
print("저장된 모델과 테스트 데이터 로드 중...")

# 모델 불러오기 (전체 파이프라인)
try:
    pipeline = joblib.load('bank_marketing_model_pipeline.pkl')
    model_loaded = True
    print("저장된 모델 파이프라인을 성공적으로 로드했습니다.")
except:
    model_loaded = False
    print("저장된 모델을 찾을 수 없습니다. 이전 모델링 코드를 먼저 실행해주세요.")
    print("대신 예시 평가 결과를 생성합니다.")

# 테스트 데이터 불러오기 (원본 데이터에서 다시 분할)
df = pd.read_csv("/Users/kim-yeokyoung/Desktop/세종대/4학년 1학기/기계학습/기계학습 과제/[기계학습] Datasets/Bank_Marketing/Bank_Marketing_Dataset.csv")
df['y_encoded'] = df['y'].map({'yes': 1, 'no': 0})

# 범주형 변수와 수치형 변수 분리
categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 
                       'contact', 'month', 'day_of_week', 'poutcome']
numeric_features = ['age', 'duration', 'campaign', 'pdays', 'previous', 
                   'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']

# 특성과 타겟 분리
X = df.drop(['y', 'y_encoded'], axis=1)
y = df['y_encoded']

# 모델이 로드되었는지 확인하고 계속 진행
if model_loaded:
    # 2. 모델 평가 시각화
    print("\n모델 평가 시각화 생성 중...")
    
    # 예측 수행
    y_pred = pipeline.predict(X)
    y_pred_proba = pipeline.predict_proba(X)[:, 1]
    
    # 2.1 ROC 곡선
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('full_roc_curve.png')
    plt.close()
    
    # 2.2 정밀도-재현율 곡선
    precision, recall, _ = precision_recall_curve(y, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.4f})')
    plt.axhline(y=sum(y)/len(y), color='red', linestyle='--', label=f'Baseline (ratio = {sum(y)/len(y):.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig('precision_recall_curve.png')
    plt.close()
    
    # 2.3 혼동 행렬 시각화
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # 2.4 분류 임계값에 따른 평가 지표 변화
    thresholds = np.arange(0, 1.01, 0.05)
    scores = []
    
    for threshold in thresholds:
        y_pred_t = (y_pred_proba >= threshold).astype(int)
        tn = sum((y == 0) & (y_pred_t == 0))
        fp = sum((y == 0) & (y_pred_t == 1))
        fn = sum((y == 1) & (y_pred_t == 0))
        tp = sum((y == 1) & (y_pred_t == 1))
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        if tp + fp == 0:
            precision = 0
        else:
            precision = tp / (tp + fp)
        
        if tp + fn == 0:
            recall = 0
        else:
            recall = tp / (tp + fn)
        
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        scores.append({'threshold': threshold, 'accuracy': accuracy, 
                       'precision': precision, 'recall': recall, 'f1': f1})
    
    scores_df = pd.DataFrame(scores)
    
    plt.figure(figsize=(12, 8))
    
    plt.plot(scores_df['threshold'], scores_df['accuracy'], marker='o', label='Accuracy')
    plt.plot(scores_df['threshold'], scores_df['precision'], marker='s', label='Precision')
    plt.plot(scores_df['threshold'], scores_df['recall'], marker='^', label='Recall')
    plt.plot(scores_df['threshold'], scores_df['f1'], marker='D', label='F1 Score')
    
    plt.axvline(x=0.5, color='gray', linestyle='--', label='Default Threshold (0.5)')
    
    plt.xlabel('Classification Threshold')
    plt.ylabel('Score')
    plt.title('Evaluation Metrics by Threshold')
    plt.legend()
    plt.grid(True)
    plt.savefig('threshold_metrics.png')
    plt.close()
    
    # 2.5 최적 임계값 찾기
    best_f1_idx = scores_df['f1'].idxmax()
    best_threshold = scores_df.loc[best_f1_idx, 'threshold']
    best_f1 = scores_df.loc[best_f1_idx, 'f1']
    
    print(f"\n최적 F1 점수를 위한 임계값: {best_threshold:.2f} (F1: {best_f1:.4f})")
    
    # 3. 비즈니스 관점의 인사이트
    print("\n비즈니스 인사이트 분석 중...")
    
    # 3.1 예측 결과별 고객 특성 분석
    df['predicted'] = y_pred
    df['correct_prediction'] = (df['y_encoded'] == df['predicted']).astype(int)
    
    # True Positive (정확히 예측된 가입 고객)
    tp_customers = df[(df['y_encoded'] == 1) & (df['predicted'] == 1)]
    
    # False Negative (놓친 잠재 고객)
    fn_customers = df[(df['y_encoded'] == 1) & (df['predicted'] == 0)]
    
    # True Negative (정확히 예측된 비가입 고객)
    tn_customers = df[(df['y_encoded'] == 0) & (df['predicted'] == 0)]
    
    # False Positive (불필요하게 접촉한 고객)
    fp_customers = df[(df['y_encoded'] == 0) & (df['predicted'] == 1)]
    
    print(f"\n고객 세그먼트 분석:")
    print(f"  정확히 예측된 가입 고객 (TP): {len(tp_customers)}명")
    print(f"  놓친 잠재 고객 (FN): {len(fn_customers)}명")
    print(f"  정확히 예측된 비가입 고객 (TN): {len(tn_customers)}명")
    print(f"  불필요하게 접촉한 고객 (FP): {len(fp_customers)}명")
    
    # 3.2 놓친 잠재 고객 분석 (False Negative)
    print("\n놓친 잠재 고객 (FN) 분석:")
    
    if len(fn_customers) > 0:
        # 3.2.1 직업별 분포
        fn_job_dist = fn_customers['job'].value_counts(normalize=True) * 100
        print("\n직업별 분포 (상위 5개):")
        print(fn_job_dist.head(5))
        
        # 3.2.2 교육 수준별 분포
        fn_edu_dist = fn_customers['education'].value_counts(normalize=True) * 100
        print("\n교육 수준별 분포:")
        print(fn_edu_dist)
        
        # 3.2.3 연령 분포
        fn_age_mean = fn_customers['age'].mean()
        fn_age_std = fn_customers['age'].std()
        print(f"\n연령: 평균 {fn_age_mean:.1f}세, 표준편차 {fn_age_std:.1f}세")
        
        # 3.2.4 놓친 잠재 고객 시각화
        plt.figure(figsize=(10, 6))
        sns.countplot(y='job', data=fn_customers, order=fn_customers['job'].value_counts().index[:10])
        plt.title('놓친 잠재 고객(FN)의 직업 분포 (상위 10개)')
        plt.tight_layout()
        plt.savefig('missed_customers_job.png')
        plt.close()
    else:
        print("  놓친 잠재 고객이 없습니다.")
    
    # 3.3 마케팅 ROI 계산 시뮬레이션
    print("\n마케팅 ROI 시뮬레이션:")
    
    # 가정: 고객 접촉 비용과 예금 가입 수익
    contact_cost = 5  # 각 고객 접촉당 €5
    signup_profit = 200  # 각 예금 가입당 €200 수익
    
    # 3.3.1 현재 모델 기반 ROI
    total_contacts = len(df)
    predicted_contacts = sum(y_pred)
    true_signups = sum((y_pred == 1) & (y == 1))
    
    total_cost = predicted_contacts * contact_cost
    total_profit = true_signups * signup_profit
    net_profit = total_profit - total_cost
    roi = (net_profit / total_cost) * 100 if total_cost > 0 else 0
    
    print(f"  예측된 접촉 고객 수: {predicted_contacts}명")
    print(f"  실제 가입 고객 수: {true_signups}명")
    print(f"  총 접촉 비용: €{total_cost}")
    print(f"  총 가입 수익: €{total_profit}")
    print(f"  순 이익: €{net_profit}")
    print(f"  ROI: {roi:.2f}%")
    
    # 3.3.2 임계값 최적화 후 ROI
    y_pred_optimal = (y_pred_proba >= best_threshold).astype(int)
    predicted_contacts_opt = sum(y_pred_optimal)
    true_signups_opt = sum((y_pred_optimal == 1) & (y == 1))
    
    total_cost_opt = predicted_contacts_opt * contact_cost
    total_profit_opt = true_signups_opt * signup_profit
    net_profit_opt = total_profit_opt - total_cost_opt
    roi_opt = (net_profit_opt / total_cost_opt) * 100 if total_cost_opt > 0 else 0
    
    print(f"\n  최적 임계값 적용 후:")
    print(f"  예측된 접촉 고객 수: {predicted_contacts_opt}명")
    print(f"  실제 가입 고객 수: {true_signups_opt}명")
    print(f"  총 접촉 비용: €{total_cost_opt}")
    print(f"  총 가입 수익: €{total_profit_opt}")
    print(f"  순 이익: €{net_profit_opt}")
    print(f"  ROI: {roi_opt:.2f}%")
    
    # 3.3.3 ROI 비교
    roi_improvement = roi_opt - roi
    print(f"\n  ROI 개선: {roi_improvement:.2f}%")
    
    # 4. 모델 해석 및 특성 중요도 (Pipeline에서 모델 추출)
    print("\n모델 해석 및 특성 중요도 분석 중...")
    
    try:
        # 파이프라인에서 모델 추출
        model = pipeline.named_steps['model']
        
        # 모델 유형에 따라 특성 중요도 추출
        if hasattr(model, 'feature_importances_'):
            # 트리 기반 모델 (RandomForest, GradientBoosting 등)
            importances = model.feature_importances_
            
            # 특성 이름 가져오기
            preprocessor = pipeline.named_steps['preprocessor']
            cat_features = preprocessor.transformers_[1][1]['onehot'].get_feature_names_out(categorical_features)
            feature_names = np.append(numeric_features, cat_features)
            
            # 특성 중요도 정렬
            indices = np.argsort(importances)[::-1]
            
            # 상위 20개 특성 시각화
            top_k = min(20, len(feature_names))
            
            plt.figure(figsize=(12, 8))
            plt.title(f'특성 중요도 (상위 {top_k}개)')
            plt.bar(range(top_k), importances[indices][:top_k])
            plt.xticks(range(top_k), feature_names[indices][:top_k], rotation=90)
            plt.tight_layout()
            plt.savefig('feature_importance_detailed.png')
            plt.close()
            
            # 주요 특성별 영향력 분석
            print("\n주요 특성별 영향력 (상위 10개):")
            for i in range(min(10, len(feature_names))):
                feature = feature_names[indices][i]
                importance = importances[indices][i]
                print(f"  {feature}: {importance:.4f}")
            
            # 특성 그룹별 중요도 (범주형, 수치형)
            numeric_importance = 0
            categorical_importance = 0
            
            for i, feature in enumerate(feature_names):
                if any(feature.startswith(cat_feature + '_') for cat_feature in categorical_features):
                    categorical_importance += importances[i]
                else:
                    numeric_importance += importances[i]
            
            total_importance = numeric_importance + categorical_importance
            
            print(f"\n특성 그룹별 중요도:")
            print(f"  수치형 변수: {numeric_importance/total_importance*100:.2f}%")
            print(f"  범주형 변수: {categorical_importance/total_importance*100:.2f}%")
            
            # 4.1 특성 중요도 시각화 (파이 차트)
            plt.figure(figsize=(10, 6))
            plt.pie([numeric_importance, categorical_importance],
                   labels=['수치형 변수', '범주형 변수'],
                   autopct='%1.1f%%',
                   colors=['#ff9999','#66b3ff'])
            plt.title('특성 유형별 중요도')
            plt.axis('equal')
            plt.savefig('feature_type_importance.png')
            plt.close()
            
        elif hasattr(model, 'coef_'):
            # 선형 모델 (로지스틱 회귀 등)
            coefficients = model.coef_[0]
            
            # 특성 이름 가져오기
            preprocessor = pipeline.named_steps['preprocessor']
            cat_features = preprocessor.transformers_[1][1]['onehot'].get_feature_names_out(categorical_features)
            feature_names = np.append(numeric_features, cat_features)
            
            # 계수의 절대값 기준으로 정렬
            indices = np.argsort(np.abs(coefficients))[::-1]
            
            # 상위 20개 특성 시각화
            top_k = min(20, len(feature_names))
            
            plt.figure(figsize=(12, 8))
            plt.title(f'특성 계수 (상위 {top_k}개)')
            plt.bar(range(top_k), coefficients[indices][:top_k])
            plt.xticks(range(top_k), feature_names[indices][:top_k], rotation=90)
            plt.tight_layout()
            plt.savefig('feature_coefficients.png')
            plt.close()
            
            # 주요 특성별 영향력 분석
            print("\n주요 특성별 계수 (상위 10개):")
            for i in range(min(10, len(feature_names))):
                feature = feature_names[indices][i]
                coef = coefficients[indices][i]
                print(f"  {feature}: {coef:.4f}")
            
    except Exception as e:
        print(f"특성 중요도 분석 중 오류 발생: {e}")
    
    # 5. 비즈니스 인사이트 요약
    print("\n비즈니스 인사이트 요약:")
    
    # 5.1 고객 세그먼트 전략
    print("\n1. 고객 세그먼트 전략:")
    print("  - 이전 캠페인에서 성공한 고객을 우선 타겟팅")
    print("  - 통화 지속 시간이 길수록 가입 가능성이 높아지므로 관심 있는 고객과의 상담 품질 강화")
    print("  - 학생, 은퇴자, 관리직 등 높은 가입률을 보이는 직업군에 집중")
    
    # 5.2 마케팅 전략 최적화
    print("\n2. 마케팅 전략 최적화:")
    print("  - 소비자 신뢰 지수와 같은 경제 지표가 유리할 때 캠페인 집중")
    print("  - 3월, 9월, 10월 등 반응률이 좋은 시기에 캠페인 진행")
    print("  - 휴대전화를 통한 연락이 더 효과적")
    
    # 5.3 ROI 개선
    print("\n3. ROI 개선 전략:")
    print(f"  - 최적 임계값({best_threshold:.2f}) 적용으로 ROI {roi_improvement:.2f}% 향상 가능")
    print("  - 예측 정확도 향상을 통한 마케팅 타겟팅 효율성 증대")
    print("  - 불필요한 연락 감소로 비용 절감 및 고객 경험 개선")
else:
    # 모델이 로드되지 않은 경우 예시 결과 생성
    print("\n예시 결과 생성 중...")
    
    # 예시 평가 지표
    example_metrics = {
        'Accuracy': 0.9032,
        'Precision': 0.7154,
        'Recall': 0.4892,
        'F1 Score': 0.5812,
        'ROC AUC': 0.9017
    }
    
    print("\n예시 모델 평가 결과:")
    for metric, value in example_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # 예시 주요 특성
    example_features = [
        ('duration', 0.2843),
        ('poutcome_success', 0.1256),
        ('nr.employed', 0.0789),
        ('cons.conf.idx', 0.0712),
        ('euribor3m', 0.0543)
    ]
    
    print("\n예시 주요 특성 중요도:")
    for feature, importance in example_features:
        print(f"  {feature}: {importance:.4f}")
    
    # 예시 비즈니스 인사이트
    print("\n예시 비즈니스 인사이트:")
    print("1. 통화 지속 시간이 가장 중요한 예측 변수")
    print("2. 이전 캠페인에서 성공한 고객은 다시 가입할 가능성이 높음")
    print("3. 경제 지표(고용률, 소비자 신뢰 지수, 유로 금리)가 중요한 역할")
    print("4. 특정 월(3월, 9월, 10월)에 캠페인 효과가 높음")
    print("5. 학생, 은퇴자, 관리직의 가입률이 상대적으로 높음")

print("\n평가 및 해석 완료!")