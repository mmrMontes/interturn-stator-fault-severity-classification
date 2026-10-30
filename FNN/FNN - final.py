import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.feature_selection import mutual_info_classif, chi2
from skfeature.function.similarity_based import fisher_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score, confusion_matrix
import torch
from torch import nn
import warnings
warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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


class FNNModel(nn.Module):
    def __init__(self, input_dim, layerdim, dropout_val=0.4):
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


kfold_results = pd.read_csv('KFOLDfnn.csv')
kfold_results = kfold_results.sort_values(
    by=['Value', 'Number'],
    ascending=[False, True]
).reset_index(drop=True)
kfold_results = kfold_results.iloc[0:1, :]

print("\n" + "="*60)
print(f"BEST K-FOLD CONFIGURATION")
print("="*60)
print(kfold_results[['Number', 'Value', 'Param k', 'Param layerdim', 'Param dropout_val', 'Param selector_value']].to_string(index=False))

results_final = []
seeds_range = range(0, 11)

print(f"\n" + "="*60)
print(f"TRAINING WITH {len(kfold_results)} CONFIGURATIONS x {len(list(seeds_range))} SEEDS = {len(kfold_results) * len(list(seeds_range))} EXPERIMENTS")
print("="*60)

for idx in range(len(kfold_results)):
    print(f"\n\n{'#'*60}")
    print(f"CONFIGURATION {idx+1}/10")
    print(f"{'#'*60}")

    k = int(kfold_results.iloc[idx]['Param k'])
    layerdim = int(kfold_results.iloc[idx]['Param layerdim'])
    dropout_val = float(kfold_results.iloc[idx]['Param dropout_val'])
    selector_value = int(kfold_results.iloc[idx]['Param selector_value'])
    val_metric = kfold_results.iloc[idx]['Value']
    config_number = int(kfold_results.iloc[idx]['Number'])

    print(f"\nParameters:")
    print(f"  k (features): {k}")
    print(f"  layerdim: {layerdim}")
    print(f"  dropout_val: {dropout_val}")
    print(f"  selector_value: {selector_value}")
    print(f"  Validation accuracy (tuning metric): {val_metric:.4f}")

    feature_selector_opc = ["variance", "info_gain", "pearson", "fisher", "chi_squared"]
    feature_selector = feature_selector_opc[selector_value]
    print(f"  Feature selector: {feature_selector}")

    for seed in seeds_range:
        print(f"\n\n{'='*60}")
        print(f"EVALUATION {idx+1}/10 - SEED {seed}")
        print(f"{'='*60}")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, shuffle=True, random_state=seed)
        print(f"\nDimensions for seed={seed}:")
        print(f"  X_train shape: {X_train.shape}")
        print(f"  X_test shape: {X_test.shape}")


        X_train_inner, X_val_inner, y_train_inner, y_val_inner = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, shuffle=True, random_state=seed)


        min_max_scaler_X = MinMaxScaler()
        X_train_norm = min_max_scaler_X.fit_transform(X_train_inner)
        X_val_norm = min_max_scaler_X.transform(X_val_inner)
        X_test_norm = min_max_scaler_X.transform(X_test)

        X_train_norm = pd.DataFrame(X_train_norm, columns=X_train_inner.columns)
        X_val_norm = pd.DataFrame(X_val_norm, columns=X_val_inner.columns)
        X_test_norm = pd.DataFrame(X_test_norm, columns=X_test.columns)


        keep_indices = list(range(len(X_train_norm)))
        for i in range(len(X_train_inner.columns)):
            outlier_idx = outlier_removal(X_train_norm.iloc[:, i].values, threshold=2.2)
            keep_indices = list(set(keep_indices) & set(outlier_idx))

        X_train_norm = X_train_norm.iloc[sorted(keep_indices), :]
        y_train_clean = y_train_inner.iloc[sorted(keep_indices)]


        if feature_selector == 'variance':
            importances = X_train_norm.var().values
        elif feature_selector == 'info_gain':
            importances = mutual_info_classif(X_train_norm, y_train_clean, random_state=seed)
        elif feature_selector == 'pearson':
            importances = X_train_norm.corrwith(y_train_clean).abs().values
        elif feature_selector == 'fisher':
            importances = fisher_score.fisher_score(X_train_norm.values, y_train_clean.values)
        elif feature_selector == 'chi_squared':
            importances = chi2(X_train_norm, y_train_clean)[0]

        feat_importances = pd.Series(importances, index=X_train_norm.columns).sort_values(ascending=False)
        top_features = list(feat_importances.index[:k])
        X_train_selected = X_train_norm[top_features].values
        X_val_selected = X_val_norm[top_features].values
        X_test_selected = X_test_norm[top_features].values


        X_class_1 = X_train_selected[y_train_clean.values == 1]
        y_class_1 = y_train_clean.values[y_train_clean.values == 1]
        X_class_2 = X_train_selected[y_train_clean.values == 2]
        y_class_2 = y_train_clean.values[y_train_clean.values == 2]
        X_class_3 = X_train_selected[y_train_clean.values == 3]
        y_class_3 = y_train_clean.values[y_train_clean.values == 3]
        X_class_4 = X_train_selected[y_train_clean.values == 4]
        y_class_4 = y_train_clean.values[y_train_clean.values == 4]

        min_samples_per_class = min(len(y_class_1), len(y_class_2), len(y_class_3), len(y_class_4))

        X_class_1_balanced = resample(X_class_1, n_samples=min_samples_per_class, random_state=seed)
        y_class_1_balanced = resample(y_class_1, n_samples=min_samples_per_class, random_state=seed)
        X_class_2_balanced = resample(X_class_2, n_samples=min_samples_per_class, random_state=seed)
        y_class_2_balanced = resample(y_class_2, n_samples=min_samples_per_class, random_state=seed)
        X_class_3_balanced = resample(X_class_3, n_samples=min_samples_per_class, random_state=seed)
        y_class_3_balanced = resample(y_class_3, n_samples=min_samples_per_class, random_state=seed)
        X_class_4_balanced = resample(X_class_4, n_samples=min_samples_per_class, random_state=seed)
        y_class_4_balanced = resample(y_class_4, n_samples=min_samples_per_class, random_state=seed)

        X_train_model = np.vstack((X_class_1_balanced, X_class_2_balanced, X_class_3_balanced, X_class_4_balanced))
        y_train_model = np.hstack((y_class_1_balanced, y_class_2_balanced, y_class_3_balanced, y_class_4_balanced))


        y_train_model_oh = convert_to_onehot(y_train_model, num_classes=4)
        y_val_oh = convert_to_onehot(y_val_inner.values, num_classes=4)

        model = FNNModel(input_dim=X_train_model.shape[1], layerdim=layerdim, dropout_val=dropout_val)
        model.to(device)

        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=50)

        X_train_tensor = torch.FloatTensor(X_train_model).to(device)
        y_train_tensor = torch.FloatTensor(y_train_model_oh).to(device)
        X_val_tensor = torch.FloatTensor(X_val_selected).to(device)
        y_val_tensor = torch.FloatTensor(y_val_oh).to(device)
        X_test_tensor = torch.FloatTensor(X_test_selected).to(device)

        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20


        for epoch in range(1500):
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

            val_loss = loss_fn(val_logits, y_val_tensor)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break


        model.eval()
        with torch.no_grad():
            y_train_pred = torch.argmax(torch.sigmoid(model(X_train_tensor)), dim=1).cpu().numpy() + 1
            y_test_pred = torch.argmax(torch.sigmoid(model(X_test_tensor)), dim=1).cpu().numpy() + 1

        acc_train = accuracy_score(y_train_model, y_train_pred)
        acc_test = accuracy_score(y_test.values, y_test_pred)
        precision_test = precision_score(y_test.values, y_test_pred, average='weighted', zero_division=0)
        recall_test = recall_score(y_test.values, y_test_pred, average='weighted', zero_division=0)
        f1_test = f1_score(y_test.values, y_test_pred, average='weighted', zero_division=0)

        with torch.no_grad():
            y_test_proba = torch.softmax(model(X_test_tensor), dim=1).cpu().numpy()

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
            'Layerdim': layerdim,
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

results_df.to_csv('FNN_results_with_seeds.csv', index=False)
print(f"\nFile saved: FNN_results_with_seeds.csv")

print('\nTOP 20 RESULTS (sorted by TEST accuracy):')
print(results_df.nlargest(20, 'Acc_Test')[['Seed', 'Configuration', 'Config_Number', 'CV_Value', 'Acc_Train', 'Acc_Test', 'Precision_Test', 'Recall_Test', 'F1_Test', 'AUC_Test', 'Balanced_Acc_Test']].to_string(index=False))


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


stats_by_config.to_csv('FNN_stats_by_config.csv')
print(f"File saved: FNN_stats_by_config.csv")


best_config_mean = results_df.groupby('Config_Number')['Acc_Test'].mean().idxmax()
best_config_data = results_df[results_df['Config_Number'] == best_config_mean]

print(f"\n{'='*80}")
print('BEST CONFIGURATION (by mean TEST accuracy)')
print(f"{'='*80}")
print(f"Config Number: {int(best_config_mean)}")
print(f"Accuracy Mean: {best_config_data['Acc_Test'].mean():.4f} ± {best_config_data['Acc_Test'].std():.4f}")
print(f"Precision Mean: {best_config_data['Precision_Test'].mean():.4f} ± {best_config_data['Precision_Test'].std():.4f}")
print(f"Recall Mean: {best_config_data['Recall_Test'].mean():.4f} ± {best_config_data['Recall_Test'].std():.4f}")
print(f"F1-Score Mean: {best_config_data['F1_Test'].mean():.4f} ± {best_config_data['F1_Test'].std():.4f}")
print(f"AUC Mean: {best_config_data['AUC_Test'].mean():.4f} ± {best_config_data['AUC_Test'].std():.4f}")
print(f"Balanced Acc Mean: {best_config_data['Balanced_Acc_Test'].mean():.4f} ± {best_config_data['Balanced_Acc_Test'].std():.4f}")
print(f"{'='*80}\n")
