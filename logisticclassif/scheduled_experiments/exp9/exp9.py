import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold, SelectKBest, chi2
from skfeature.function.similarity_based import fisher_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score
import warnings
warnings.filterwarnings('ignore')


def convertObj_to_Int(data):
    data_int = data.copy()
    for i in range(data.shape[1]):
        if data_int.iloc[:, i].dtype == 'O':
            data_int.iloc[:, i] = data_int.iloc[:, i].astype('category').cat.codes
    return data_int.astype('int')


def outlier_removal(feature, threshold):
    miu = feature.mean()
    sigma = feature.std()
    features_mantained = [i for i in range(len(feature)) if abs(feature[i] - miu) < (sigma * threshold)]
    return features_mantained


data = pd.read_excel('C:\\Users\\montes\\OneDrive\\DPI\\2026-1\\paper Leonel\\Random_Forest_Info_Gain\\Random_Forest_Info_Gain\\github\\Data_IBARRAM.xlsx')

data_int = convertObj_to_Int(data)
data_int = data_int.dropna()
X = data_int.drop('Fault', axis=1)
y = data_int.Fault


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, shuffle=True, random_state=0)


kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)


k = 50.0
C = 100.0
selector_value = 0.0
penalty_value = 0.0


penalty_value = int(penalty_value)
if penalty_value == 0:
    penalty = 'l1'
elif penalty_value == 1:
    penalty = 'l2'


k = int(k)
selector_value = int(selector_value)
max_iter = 1500
feature_selector_opc = ["variance", "info_gain", "pearson", "fisher", "chi_squared"]
feature_selector = feature_selector_opc[selector_value]


acc_train_fold = []
acc_val_fold = []
precision_train_fold = []
recall_train_fold = []
f1_train_fold = []
precision_val_fold = []
recall_val_fold = []
f1_val_fold = []
auc_train_fold = []
auc_val_fold = []
specificity_train_fold = []
specificity_val_fold = []
balanced_acc_train_fold = []
balanced_acc_val_fold = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train), start=1):


  X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
  y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]


  min_max_scaler_X = MinMaxScaler()
  X_train_norm = min_max_scaler_X.fit_transform(X_train_fold)
  X_val_norm = min_max_scaler_X.transform(X_val_fold)

  X_train_norm = pd.DataFrame(X_train_norm, columns=X_train.columns)
  X_val_norm = pd.DataFrame(X_val_norm, columns=X_val_fold.columns)


  for i in range(len(X_train_norm.columns)):
    keep_idx = outlier_removal(X_train_norm.iloc[:, i].values, threshold=2.2)
    X_train_norm = X_train_norm.iloc[keep_idx, :]
    y_train_fold = y_train_fold.iloc[keep_idx]


  if feature_selector == 'variance':
    importances = X_train_norm.var().values
    feat_importances = pd.Series(importances, index=X_train_norm.columns).sort_values(ascending=False)
    top_features = list(feat_importances.index[:k])
    X_train_selected = X_train_norm[top_features].values
    X_val_selected = X_val_norm[top_features].values


  elif feature_selector == 'info_gain':
    importances = mutual_info_classif(X_train_norm, y_train_fold, random_state=0)
    feat_importances = pd.Series(importances, index=X_train_norm.columns).sort_values(ascending=False)
    top_features = list(feat_importances.index[:k])
    X_train_selected = X_train_norm[top_features].values
    X_val_selected = X_val_norm[top_features].values


  elif feature_selector == 'pearson':
    importances = X_train_norm.corrwith(y_train_fold).abs().values
    feat_importances = pd.Series(importances, index=X_train_norm.columns).sort_values(ascending=False)
    top_features = list(feat_importances.index[:k])
    X_train_selected = X_train_norm[top_features].values
    X_val_selected = X_val_norm[top_features].values


  elif feature_selector == 'fisher':
    importances = fisher_score.fisher_score(X_train_norm.values, y_train_fold.values)
    feat_importances = pd.Series(importances, index=X_train_norm.columns).sort_values(ascending=False)
    top_features = list(feat_importances.index[:k])
    X_train_selected = X_train_norm[top_features].values
    X_val_selected = X_val_norm[top_features].values


  elif feature_selector == 'chi_squared':
    importances = chi2(X_train_norm, y_train_fold)[0]
    feat_importances = pd.Series(importances, index=X_train_norm.columns).sort_values(ascending=False)
    top_features = list(feat_importances.index[:k])
    X_train_selected = X_train_norm[top_features].values
    X_val_selected = X_val_norm[top_features].values


  X_class_1 = X_train_selected[y_train_fold.values == 1]
  y_class_1 = y_train_fold.values[y_train_fold.values == 1]
  X_class_2 = X_train_selected[y_train_fold.values == 2]
  y_class_2 = y_train_fold.values[y_train_fold.values == 2]
  X_class_3 = X_train_selected[y_train_fold.values == 3]
  y_class_3 = y_train_fold.values[y_train_fold.values == 3]
  X_class_4 = X_train_selected[y_train_fold.values == 4]
  y_class_4 = y_train_fold.values[y_train_fold.values == 4]

  min_samples_per_class = min(len(y_class_1), len(y_class_2), len(y_class_3), len(y_class_4))

  X_class_1_balanced = resample(X_class_1, n_samples=min_samples_per_class, random_state=0)
  y_class_1_balanced = resample(y_class_1, n_samples=min_samples_per_class, random_state=0)
  X_class_2_balanced = resample(X_class_2, n_samples=min_samples_per_class, random_state=0)
  y_class_2_balanced = resample(y_class_2, n_samples=min_samples_per_class, random_state=0)
  X_class_3_balanced = resample(X_class_3, n_samples=min_samples_per_class, random_state=0)
  y_class_3_balanced = resample(y_class_3, n_samples=min_samples_per_class, random_state=0)
  X_class_4_balanced = resample(X_class_4, n_samples=min_samples_per_class, random_state=0)
  y_class_4_balanced = resample(y_class_4, n_samples=min_samples_per_class, random_state=0)


  X_train_model = np.vstack((X_class_1_balanced, X_class_2_balanced, X_class_3_balanced, X_class_4_balanced))
  y_train_model = np.hstack((y_class_1_balanced, y_class_2_balanced, y_class_3_balanced, y_class_4_balanced))
  X_val_model = X_val_selected
  y_val_model = y_val_fold.values


  solver = 'liblinear'
  clf = LogisticRegression(max_iter=max_iter, C=C, penalty=penalty, solver=solver, random_state=0)
  clf.fit(X_train_model, y_train_model)

  y_train_pred = clf.predict(X_train_model)
  y_val_pred = clf.predict(X_val_model)

  acc_train = accuracy_score(y_train_model, y_train_pred)
  acc_val = accuracy_score(y_val_model, y_val_pred)

  acc_train_fold.append(acc_train)
  acc_val_fold.append(acc_val)


  precision_train = precision_score(y_train_model, y_train_pred, average='weighted', zero_division=0)
  recall_train = recall_score(y_train_model, y_train_pred, average='weighted', zero_division=0)
  f1_train = f1_score(y_train_model, y_train_pred, average='weighted', zero_division=0)

  precision_val = precision_score(y_val_model, y_val_pred, average='weighted', zero_division=0)
  recall_val = recall_score(y_val_model, y_val_pred, average='weighted', zero_division=0)
  f1_val = f1_score(y_val_model, y_val_pred, average='weighted', zero_division=0)

  precision_train_fold.append(precision_train)
  recall_train_fold.append(recall_train)
  f1_train_fold.append(f1_train)
  precision_val_fold.append(precision_val)
  recall_val_fold.append(recall_val)
  f1_val_fold.append(f1_val)


  y_train_proba = clf.predict_proba(X_train_model)
  y_val_proba = clf.predict_proba(X_val_model)

  auc_train = roc_auc_score(y_train_model, y_train_proba, average='weighted', multi_class='ovr')
  auc_val = roc_auc_score(y_val_model, y_val_proba, average='weighted', multi_class='ovr')


  specificity_train = recall_score(y_train_model, y_train_pred, average='macro', zero_division=0)
  specificity_val = recall_score(y_val_model, y_val_pred, average='macro', zero_division=0)


  balanced_acc_train = balanced_accuracy_score(y_train_model, y_train_pred)
  balanced_acc_val = balanced_accuracy_score(y_val_model, y_val_pred)

  auc_train_fold.append(auc_train)
  auc_val_fold.append(auc_val)
  specificity_train_fold.append(specificity_train)
  specificity_val_fold.append(specificity_val)
  balanced_acc_train_fold.append(balanced_acc_train)
  balanced_acc_val_fold.append(balanced_acc_val)


acc_train_mean = np.mean(acc_train_fold)
acc_val_mean = np.mean(acc_val_fold)
precision_train_mean = np.mean(precision_train_fold)
recall_train_mean = np.mean(recall_train_fold)
f1_train_mean = np.mean(f1_train_fold)
precision_val_mean = np.mean(precision_val_fold)
recall_val_mean = np.mean(recall_val_fold)
f1_val_mean = np.mean(f1_val_fold)
auc_train_mean = np.mean(auc_train_fold)
auc_val_mean = np.mean(auc_val_fold)
specificity_train_mean = np.mean(specificity_train_fold)
specificity_val_mean = np.mean(specificity_val_fold)
balanced_acc_train_mean = np.mean(balanced_acc_train_fold)
balanced_acc_val_mean = np.mean(balanced_acc_val_fold)

print(f"Finished:{0}")
print(f"Mean Accuracy Train:{acc_train_mean}")
print(f"Mean Accuracy Validation:{acc_val_mean}")
print(f"Mean Precision Train:{precision_train_mean}")
print(f"Mean Precision Validation:{precision_val_mean}")
print(f"Mean Recall Train:{recall_train_mean}")
print(f"Mean Recall Validation:{recall_val_mean}")
print(f"Mean F1 Train:{f1_train_mean}")
print(f"Mean F1 Validation:{f1_val_mean}")
print(f"Mean AUC Train:{auc_train_mean}")
print(f"Mean AUC Validation:{auc_val_mean}")
print(f"Mean Specificity Train:{specificity_train_mean}")
print(f"Mean Specificity Validation:{specificity_val_mean}")
print(f"Mean Balanced Accuracy Train:{balanced_acc_train_mean}")
print(f"Mean Balanced Accuracy Validation:{balanced_acc_val_mean}")


