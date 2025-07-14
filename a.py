import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# تحميل البيانات
df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
X = df.drop('DEATH_EVENT', axis=1)
y = df['DEATH_EVENT']

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                    test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SMOTE
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_train_scaled, y_train)

# Feature Selection (خليهم كلهم المرة دي)
selector = SelectKBest(score_func=f_classif, k='all')
X_resampled_fs = selector.fit_transform(X_resampled, y_resampled)
X_test_fs = selector.transform(X_test_scaled)

# MLP محسن
mlp_model = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu',
                          solver='adam', max_iter=800, random_state=42)

# SVC محسن
svc_model = SVC(kernel='rbf', probability=True, C=2.0, gamma='auto', random_state=42)

# XGBoost
xgb_model = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                              use_label_encoder=False, eval_metric='logloss', random_state=42)

# Voting Classifier باستخدام أفضل 3 موديلات
voting = VotingClassifier(estimators=[
    ('mlp', mlp_model),
    ('svc', svc_model),
    ('xgb', xgb_model)
], voting='soft')

# تدريب
voting.fit(X_resampled_fs, y_resampled)

# تنبؤ وتقييم
y_pred = voting.predict(X_test_fs)
acc = accuracy_score(y_test, y_pred)

print(f"\n✅ Accuracy: {acc*100:.2f}%")
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
