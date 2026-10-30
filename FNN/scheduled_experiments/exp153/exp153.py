import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.feature_selection import mutual_info_classif, chi2
from skfeature.function.similarity_based import fisher_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score
import torch
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def convert_to_onehot(y, num_classes=4):
    pass
    eye_matrix = np.eye(num_classes)

    return eye_matrix[y.astype(int) - 1]

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

def accuracy_fn(y_true, y_pred):
    return (y_true == y_pred).sum().item() / len(y_true)


class CircleModelV2(nn.Module):
      def __init__(self, input_dim, layerdim, dropout_val = 0.4):
          super().__init__()
          self.layer_1 = nn.Linear(input_dim, layerdim)
          self.dropout1 = nn.Dropout(dropout_val)
          self.layer_2 = nn.Linear(layerdim, layerdim // 2)
          self.dropout2 = nn.Dropout(dropout_val)
          self.layer_3 = nn.Linear(layerdim // 2, layerdim // 4)
          self.dropout3 = nn.Dropout(dropout_val)
          self.layer_4 = nn.Linear(layerdim // 4, 4)
          self.relu = nn.ReLU()

      def forward(self, x):
          x = self.relu(self.layer_1(x))
          x = self.dropout1(x)
          x = self.relu(self.layer_2(x))
          x = self.dropout2(x)
          x = self.relu(self.layer_3(x))
          x = self.dropout3(x)
          x = self.layer_4(x)
          return x


data = pd.read_excel('C:\\Users\\montes\\OneDrive\\DPI\\2026-1\\paper Leonel\\Random_Forest_Info_Gain\\Random_Forest_Info_Gain\\github\\Data_IBARRAM.xlsx')

data_int = convertObj_to_Int(data)
data_int = data_int.dropna()
X = data_int.drop('Fault', axis=1)
y = data_int.Fault


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, shuffle=True, random_state=0)


kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)


k = 500.0
layerdim = 32.0
selector_value = 1.0
dropout_val = 0.4

learning_rate = 0.001
epochs = 1500
k = int(k)
layerdim = int(layerdim)
epochs = int(epochs)
selector_value = int(selector_value)

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
  y_train_model_raw = np.hstack((y_class_1_balanced, y_class_2_balanced, y_class_3_balanced, y_class_4_balanced))
  y_train_model = convert_to_onehot(y_train_model_raw, num_classes=4)
  X_val_model = X_val_selected
  y_val_model = convert_to_onehot(y_val_fold.values, num_classes=4)


  model = CircleModelV2(input_dim=X_val_model.shape[1],layerdim=layerdim,dropout_val=dropout_val).to(device)
  loss_fn = nn.BCEWithLogitsLoss()
  optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=50)

  best_val_loss = float('inf')
  patience_counter = 0
  patience = 20

  for epoch in range(epochs):
    model.train()
    logits = model(torch.tensor(X_train_model, dtype=torch.float32).to(device))
    y_pred_probs = torch.sigmoid(logits)
    y_pred = torch.argmax(y_pred_probs, dim=1)
    loss = loss_fn(logits, torch.tensor(y_train_model, dtype=torch.float32).to(device))
    y_train_true = torch.argmax(torch.tensor(y_train_model, dtype=torch.float32).to(device), dim=1)
    acc = accuracy_fn(y_true=y_train_true, y_pred=y_pred)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        val_logits = model(torch.tensor(X_val_model, dtype=torch.float32).to(device))
        val_preds_probs = torch.sigmoid(val_logits)
        val_preds = torch.argmax(val_preds_probs, dim=1)
        y_val_true = torch.argmax(torch.tensor(y_val_model, dtype=torch.float32).to(device), dim=1)
        val_acc = accuracy_fn(y_true=y_val_true, y_pred=val_preds)

    scheduler.step(val_acc)


    val_loss = loss_fn(val_logits, torch.tensor(y_val_model, dtype=torch.float32).to(device))
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      patience_counter = 0
    else:
      patience_counter += 1
      if patience_counter >= patience:
        break

  y_train_logits = model(torch.tensor(X_train_model, dtype=torch.float32).to(device))
  y_train_pred = torch.argmax(torch.sigmoid(y_train_logits), dim=1).cpu().numpy()
  y_val_logits = model(torch.tensor(X_val_model, dtype=torch.float32).to(device))
  y_val_pred = torch.argmax(torch.sigmoid(y_val_logits), dim=1).cpu().numpy()

  y_train_true = np.argmax(y_train_model, axis=1)
  y_val_true = np.argmax(y_val_model, axis=1)

  acc_train = accuracy_score(y_train_true, y_train_pred)
  acc_val = accuracy_score(y_val_true, y_val_pred)

  acc_train_fold.append(acc_train)
  acc_val_fold.append(acc_val)


  precision_train = precision_score(y_train_true, y_train_pred, average='weighted', zero_division=0)
  recall_train = recall_score(y_train_true, y_train_pred, average='weighted', zero_division=0)
  f1_train = f1_score(y_train_true, y_train_pred, average='weighted', zero_division=0)

  precision_val = precision_score(y_val_true, y_val_pred, average='weighted', zero_division=0)
  recall_val = recall_score(y_val_true, y_val_pred, average='weighted', zero_division=0)
  f1_val = f1_score(y_val_true, y_val_pred, average='weighted', zero_division=0)

  precision_train_fold.append(precision_train)
  recall_train_fold.append(recall_train)
  f1_train_fold.append(f1_train)
  precision_val_fold.append(precision_val)
  recall_val_fold.append(recall_val)
  f1_val_fold.append(f1_val)


  model.eval()
  with torch.inference_mode():
    y_train_proba = torch.softmax(torch.sigmoid(y_train_logits.detach().clone()), dim=1).cpu().numpy()
    y_val_proba = torch.softmax(torch.sigmoid(y_val_logits.detach().clone()), dim=1).cpu().numpy()

  auc_train = roc_auc_score(y_train_true + 1, y_train_proba, average='weighted', multi_class='ovr')
  auc_val = roc_auc_score(y_val_true + 1, y_val_proba, average='weighted', multi_class='ovr')


  specificity_train = recall_score(y_train_true, y_train_pred, average='macro', zero_division=0)
  specificity_val = recall_score(y_val_true, y_val_pred, average='macro', zero_division=0)


  y_train_true_labels = y_train_true + 1
  y_val_true_labels = y_val_true + 1
  y_train_pred_labels = y_train_pred + 1
  y_val_pred_labels = y_val_pred + 1
  balanced_acc_train = balanced_accuracy_score(y_train_true_labels, y_train_pred_labels)
  balanced_acc_val = balanced_accuracy_score(y_val_true_labels, y_val_pred_labels)

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


