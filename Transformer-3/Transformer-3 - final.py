import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score, confusion_matrix, classification_report
import torch
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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


def convertObj_to_Int(data):
    data_int = data.copy()
    for i in range(data.shape[1]):
        if data_int.iloc[:, i].dtype == 'O':
            data_int.iloc[:, i] = data_int.iloc[:, i].astype('category').cat.codes
    return data_int.astype('int')

def convert_to_onehot(y, num_classes=4):
    eye_matrix = np.eye(num_classes)
    return eye_matrix[y.astype(int) - 1]

def accuracy_fn(y_true, y_pred):
    return (y_true == y_pred).sum().item() / len(y_true)
def outlier_removal(feature, threshold):
    miu = feature.mean()
    sigma = feature.std()
    features_mantained = [i for i in range(len(feature)) if abs(feature[i] - miu) < (sigma * threshold)]
    return features_mantained

def multiclass_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    specificities = []
    for i in range(cm.shape[0]):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - TP - FP - FN
        specificities.append(TN / (TN + FP) if (TN + FP) > 0 else 0.0)
    return np.mean(specificities)

class CircleModelTransformer(nn.Module):
    def __init__(self, layerdim, dropout_val=0.4, input_points=30):
        super().__init__()
        self.input_points = input_points
        self.input_projection = nn.Linear(3, layerdim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=layerdim,
            nhead=8 if layerdim % 8 == 0 else 4 if layerdim % 4 == 0 else 2,
            dim_feedforward=layerdim * 2,
            dropout=dropout_val,
            batch_first=True,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(layerdim, layerdim // 2)
        self.dropout = nn.Dropout(dropout_val)
        self.fc2 = nn.Linear(layerdim // 2, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x[:, :, :self.input_points]
        x = x.transpose(1, 2)
        x = self.input_projection(x)
        x = self.transformer_encoder(x)
        x = x.transpose(1, 2)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
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


kfold_results = pd.read_csv('KFOLDTransformer-3.csv')
kfold_results = kfold_results.sort_values(
    by=['Value', 'Number'],
    ascending=[False, True]
).reset_index(drop=True)
kfold_results = kfold_results.iloc[0:1, :]

print("\n" + "="*60)
print(f"BEST K-FOLD CONFIGURATION - TRANSFORMER-3")
print("="*60)
print(kfold_results[['Number', 'Value', 'Param layerdim', 'Param dropout_val', 'Param input_points']].to_string(index=False))

results_final = []
seeds_range = range(0, 11)

print(f"\n" + "="*60)
print(f"TRAINING WITH {len(kfold_results)} CONFIGURATIONS x {len(list(seeds_range))} SEEDS = {len(kfold_results) * len(list(seeds_range))} EXPERIMENTS")
print("="*60)

best_X_test_saved = None
best_y_test_saved = None

for idx in range(len(kfold_results)):
    print(f"\n\n{'#'*60}")
    print(f"CONFIGURATION {idx+1}/{len(kfold_results)}")
    print(f"{'#'*60}")

    trans_layerdim = int(kfold_results.iloc[idx]['Param layerdim'])
    dropout_val = float(kfold_results.iloc[idx]['Param dropout_val'])
    input_points = int(kfold_results.iloc[idx]['Param input_points'])
    val_metric = kfold_results.iloc[idx]['Value']
    config_number = int(kfold_results.iloc[idx]['Number'])

    print(f"\nParameters:")
    print(f"  transformer_layerdim: {trans_layerdim}")
    print(f"  dropout_val: {dropout_val}")
    print(f"  input_points: {input_points}")
    print(f"  Validation accuracy (tuning metric): {val_metric:.4f}")

    for seed in seeds_range:
        print(f"\n\n{'='*60}")
        print(f"EVALUATION {idx+1}/{len(kfold_results)} - SEED {seed}")
        print(f"{'='*60}")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, shuffle=True, random_state=seed)


        X_train_inner, X_val_inner, y_train_inner, y_val_inner = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, shuffle=True, random_state=seed)


        scaler_A = MinMaxScaler()
        scaler_B = MinMaxScaler()
        scaler_C = MinMaxScaler()

        A_train = scaler_A.fit_transform(X_train_inner[cols_A].values)
        B_train = scaler_B.fit_transform(X_train_inner[cols_B].values)
        C_train = scaler_C.fit_transform(X_train_inner[cols_C].values)

        A_val = scaler_A.transform(X_val_inner[cols_A].values)
        B_val = scaler_B.transform(X_val_inner[cols_B].values)
        C_val = scaler_C.transform(X_val_inner[cols_C].values)

        A_test = scaler_A.transform(X_test[cols_A].values)
        B_test = scaler_B.transform(X_test[cols_B].values)
        C_test = scaler_C.transform(X_test[cols_C].values)


        y_train_clean = y_train_inner


        X_train_model = np.stack([A_train, B_train, C_train], axis=1)
        X_val_trans = np.stack([A_val, B_val, C_val], axis=1)
        X_test_trans = np.stack([A_test, B_test, C_test], axis=1)


        y_train_model = y_train_clean.values


        X_train_filtered, y_train_filtered = remove_training_outliers(X_train_model, y_train_model)
        X_train_model, y_train_model = balance_training_classes(X_train_filtered, y_train_filtered, random_state=seed)
        y_train_model_oh = convert_to_onehot(y_train_model, num_classes=4)
        y_val_oh = convert_to_onehot(y_val_inner.values, num_classes=4)

        model = CircleModelTransformer(layerdim=trans_layerdim, dropout_val=dropout_val, input_points=input_points)
        model.to(device)

        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3000)

        X_train_tensor = torch.FloatTensor(X_train_model).to(device)
        y_train_tensor = torch.FloatTensor(y_train_model_oh).to(device)
        X_val_tensor = torch.FloatTensor(X_val_trans).to(device)
        y_val_tensor = torch.FloatTensor(y_val_oh).to(device)
        X_test_tensor = torch.FloatTensor(X_test_trans).to(device)

        best_val_loss = float('inf')
        patience_counter = 0
        patience = 7000
        best_model_state = None


        for epoch in range(10000):
            model.train()
            optimizer.zero_grad()
            logits = model(X_train_tensor)
            loss = loss_fn(logits, y_train_tensor)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.inference_mode():
                val_logits = model(X_val_tensor)
                val_preds = torch.argmax(torch.sigmoid(val_logits), dim=1)
                y_val_true = torch.argmax(y_val_tensor, dim=1)
                val_acc = accuracy_fn(y_true=y_val_true, y_pred=val_preds)

            scheduler.step(val_acc)

            val_loss = loss_fn(val_logits, y_val_tensor).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break


        if best_model_state is None:
            raise RuntimeError(f"No valid checkpoint was obtained for seed={seed}.")
        model.load_state_dict(best_model_state)


        model.eval()
        with torch.no_grad():
            y_train_pred = torch.argmax(torch.sigmoid(model(X_train_tensor)), dim=1).cpu().numpy() + 1
            y_test_pred = torch.argmax(torch.sigmoid(model(X_test_tensor)), dim=1).cpu().numpy() + 1

        acc_train = accuracy_score(y_train_model, y_train_pred)
        acc_test = accuracy_score(y_test.values, y_test_pred)
        precision_test = precision_score(y_test.values, y_test_pred, average='weighted', zero_division=0)
        recall_test = recall_score(y_test.values, y_test_pred, average='weighted', zero_division=0)
        f1_test = f1_score(y_test.values, y_test_pred, average='weighted', zero_division=0)


        if seed == 0:
            checkpoint = {
                'state_dict': model.state_dict(),
                'layerdim': trans_layerdim,
                'dropout_val': dropout_val,
                'input_points': input_points,
                'seed': seed
            }
            torch.save(checkpoint, 'best_transformer3_model.pth')
            best_X_test_saved = X_test_tensor.clone()
            best_y_test_saved = y_test.values.copy()

        with torch.no_grad():
            y_test_proba = torch.softmax(torch.sigmoid(model(X_test_tensor)), dim=1).cpu().numpy()

        try:
            auc_test = roc_auc_score(y_test.values, y_test_proba, average='weighted', multi_class='ovr')
        except:
            auc_test = 0.0

        specificity_test = multiclass_specificity(y_test.values, y_test_pred)
        balanced_acc_test = balanced_accuracy_score(y_test.values, y_test_pred)

        results_final.append({
            'Seed': seed,
            'Configuration': idx + 1,
            'Config_Number': config_number,
            'CV_Value': val_metric,
            'TransLayerdim': trans_layerdim,
            'Dropout': dropout_val,
            'Acc_Train': acc_train,
            'Acc_Test': acc_test,
            'Precision_Test': precision_test,
            'Recall_Test': recall_test,
            'F1_Test': f1_test,
            'AUC_Test': auc_test,
            'Specificity_Test': specificity_test,
            'Balanced_Acc_Test': balanced_acc_test
        })

        print(f"\n  TEST results: Acc={acc_test:.4f} | Prec={precision_test:.4f} | Rec={recall_test:.4f} | F1={f1_test:.4f}")


results_df = pd.DataFrame(results_final)
print(f"\n\n{'='*80}")
print(f"FINAL SUMMARY: {len(kfold_results)} CONFIGURATIONS x {len(list(seeds_range))} SEEDS = {len(results_df)} EXPERIMENTS")
print(f"{'='*80}")

results_df.to_csv('Transformer3_results_with_seeds.csv', index=False)
print(f"\nFile saved: Transformer3_results_with_seeds.csv")

print('\nRESULTS BY SEED (robustness; no selection using TEST):')
print(results_df.sort_values('Seed')[['Seed', 'Configuration', 'Config_Number', 'CV_Value', 'Acc_Train', 'Acc_Test', 'Precision_Test', 'Recall_Test', 'F1_Test', 'AUC_Test', 'Balanced_Acc_Test']].to_string(index=False))


print(f"\n\n{'='*80}")
print('STATISTICS BY CONFIGURATION (mean and standard deviation across seeds)')
print(f"{'='*80}")
stats_by_config = results_df.groupby('Config_Number').agg({
    'Acc_Test': ['mean', 'std'],
    'Precision_Test': ['mean', 'std'],
    'Recall_Test': ['mean', 'std'],
    'F1_Test': ['mean', 'std'],
    'AUC_Test': ['mean', 'std'],
    'Balanced_Acc_Test': ['mean', 'std']
}).round(4)
print(stats_by_config)


stats_by_config.to_csv('Transformer3_stats_by_config.csv')
print(f"File saved: Transformer3_stats_by_config.csv")


best_config_mean = config_number
best_config_data = results_df[results_df['Config_Number'] == best_config_mean]

print(f"\n{'='*80}")
print('CONFIGURATION PREVIOUSLY SELECTED BY K-FOLD/OPTUNA')
print(f"{'='*80}")
print(f"Config Number: {int(best_config_mean)}")
print(f"Accuracy Mean: {best_config_data['Acc_Test'].mean():.4f} ± {best_config_data['Acc_Test'].std():.4f}")
print(f"Precision Mean: {best_config_data['Precision_Test'].mean():.4f} ± {best_config_data['Precision_Test'].std():.4f}")
print(f"Recall Mean: {best_config_data['Recall_Test'].mean():.4f} ± {best_config_data['Recall_Test'].std():.4f}")
print(f"F1-Score Mean: {best_config_data['F1_Test'].mean():.4f} ± {best_config_data['F1_Test'].std():.4f}")
print(f"AUC Mean: {best_config_data['AUC_Test'].mean():.4f} ± {best_config_data['AUC_Test'].std():.4f}")
print(f"Balanced Acc Mean: {best_config_data['Balanced_Acc_Test'].mean():.4f} ± {best_config_data['Balanced_Acc_Test'].std():.4f}")
print(f"{'='*80}\n")


print(f"\n{'='*80}")
print('EVALUATION OF THE SAVED REFERENCE MODEL (SEED 0)')
print(f"{'='*80}")

checkpoint = torch.load('best_transformer3_model.pth', map_location=device, weights_only=True)
best_model_loaded = CircleModelTransformer(
    layerdim=checkpoint['layerdim'],
    dropout_val=checkpoint['dropout_val'],
    input_points=checkpoint['input_points']
)
best_model_loaded.load_state_dict(checkpoint['state_dict'])
best_model_loaded.to(device)
best_model_loaded.eval()

with torch.no_grad():
    y_best_pred = torch.argmax(torch.sigmoid(best_model_loaded(best_X_test_saved)), dim=1).cpu().numpy() + 1


target_names = ['10% Pole (2.5% Phase)', '20% Pole (5.0% Phase)', '30% Pole (7.5% Phase)', '40% Pole (10.0% Phase)']


report_dict = classification_report(best_y_test_saved, y_best_pred, labels=[1, 2, 3, 4], target_names=target_names, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_excel('Best_Model_Transformer3_Classification_Report.xlsx', index=True)
print('Classification report successfully saved as: Best_Model_Transformer3_Classification_Report.xlsx')

print("\nClassification Report:")
print(classification_report(best_y_test_saved, y_best_pred, labels=[1, 2, 3, 4], target_names=target_names, zero_division=0))


cm = confusion_matrix(best_y_test_saved, y_best_pred, labels=[1, 2, 3, 4])
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix - Reference Transformer-3 Model (Seed 0, ITSC Faults)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()

plt.savefig('Confusion_Matrix_Best_Model_Transformer3.png', dpi=300)
print('Confusion matrix plot saved as: Confusion_Matrix_Best_Model_Transformer3.png')
print(f"{'='*80}\n")
plt.show()
