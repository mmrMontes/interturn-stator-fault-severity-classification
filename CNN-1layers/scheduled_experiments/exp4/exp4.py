from sklearn.utils import resample
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score
import torch
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def remove_training_outliers(features, labels, threshold=2.2):
    features = np.asarray(features)
    labels = np.asarray(labels)
    flat = features.reshape(len(features), -1)
    keep_indices = np.arange(len(features))
    for column in range(flat.shape[1]):
        keep_local = outlier_removal(flat[keep_indices, column], threshold)
        keep_indices = keep_indices[keep_local]
        if len(keep_indices) == 0:
            raise ValueError('Outlier removal left no training samples.')
    return features[keep_indices], labels[keep_indices]


def balance_training_classes(features, labels, random_state):
    features = np.asarray(features)
    labels = np.asarray(labels)
    class_labels = (1, 2, 3, 4)
    min_samples_per_class = min(np.count_nonzero(labels == value) for value in class_labels)
    if min_samples_per_class == 0:
        raise ValueError('Class balancing requires training samples from all four classes.')
    balanced_features = []
    balanced_labels = []
    for value in class_labels:
        class_features = features[labels == value]
        class_targets = labels[labels == value]
        sampled_features, sampled_targets = resample(
            class_features, class_targets, n_samples=min_samples_per_class,
            random_state=random_state
        )
        balanced_features.append(sampled_features)
        balanced_labels.append(sampled_targets)
    return np.concatenate(balanced_features, axis=0), np.concatenate(balanced_labels)


def convert_to_onehot(y, num_classes=4):
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


class CircleModelCNNV1(nn.Module):
      def __init__(self, layerdim, dropout_val = 0.2, kernel_size=16):
          super().__init__()
          k1 = kernel_size

          self.conv1 = nn.Conv1d(in_channels=3, out_channels=layerdim, kernel_size=k1, padding=k1//2)
          self.dropout1 = nn.Dropout(dropout_val)
          self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

          self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

          self.fc1 = nn.Linear(layerdim, layerdim//2)
          self.dropout2 = nn.Dropout(dropout_val)
          self.fc2 = nn.Linear(layerdim//2, 4)
          self.relu = nn.ReLU()

      def forward(self, x):
          x = self.relu(self.conv1(x))
          x = self.dropout1(x)
          x = self.pool1(x)

          x = self.adaptive_pool(x)
          x = x.view(x.size(0), -1)

          x = self.relu(self.fc1(x))
          x = self.dropout2(x)
          x = self.fc2(x)
          return x


data = pd.read_excel('C:\\Users\\montes\\OneDrive\\DPI\\2026-1\\paper Leonel\\Random_Forest_Info_Gain\\Random_Forest_Info_Gain\\github\\Data_IBARRAM.xlsx')
data_int = convertObj_to_Int(data)
data_int = data_int.dropna()
X = data_int.drop('Fault', axis=1)
y = data_int.Fault


cols_A = [f'A{i}' for i in range(1, 501)]
cols_B = [f'B{i}' for i in range(1, 501)]
cols_C = [f'C{i}' for i in range(1, 501)]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, shuffle=True, random_state=0)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)


layerdim = 48.0
ker_size = 8.0
learning_rate = 0.001
dropout_val = 0.2
epochs = 7000

layerdim = int(layerdim)
ker_size = int(ker_size)
epochs = int(epochs)

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


  scaler_A = MinMaxScaler()
  scaler_B = MinMaxScaler()
  scaler_C = MinMaxScaler()

  A_train = scaler_A.fit_transform(X_train_fold[cols_A].values)
  B_train = scaler_B.fit_transform(X_train_fold[cols_B].values)
  C_train = scaler_C.fit_transform(X_train_fold[cols_C].values)

  A_val = scaler_A.transform(X_val_fold[cols_A].values)
  B_val = scaler_B.transform(X_val_fold[cols_B].values)
  C_val = scaler_C.transform(X_val_fold[cols_C].values)


  X_train_model = np.stack([A_train, B_train, C_train], axis=1)
  X_val_model = np.stack([A_val, B_val, C_val], axis=1)

  X_train_filtered, y_train_filtered = remove_training_outliers(X_train_model, y_train_fold.values)
  X_train_model, y_train_model_raw = balance_training_classes(X_train_filtered, y_train_filtered, random_state=0)
  y_train_model = convert_to_onehot(y_train_model_raw, num_classes=4)
  y_val_model = convert_to_onehot(y_val_fold.values, num_classes=4)


  model = CircleModelCNNV1(layerdim=layerdim, dropout_val=dropout_val, kernel_size=ker_size).to(device)

  patience = 4000
  loss_fn = nn.BCEWithLogitsLoss()
  optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3000)

  best_val_loss = float('inf')
  patience_counter = 0
  best_model_state = None

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
        val_loss = loss_fn(val_logits, torch.tensor(y_val_model, dtype=torch.float32).to(device))

    scheduler.step(val_acc)


    if val_loss < best_val_loss:
      best_val_loss = val_loss
      best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
      patience_counter = 0
    else:
      patience_counter += 1
      if patience_counter >= patience:
        print(f"Fold {fold} - Early stopping at epoch {epoch}")
        break


  model.load_state_dict(best_model_state)

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

  balanced_acc_train = balanced_accuracy_score(y_train_true + 1, y_train_pred + 1)
  balanced_acc_val = balanced_accuracy_score(y_val_true + 1, y_val_pred + 1)

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
