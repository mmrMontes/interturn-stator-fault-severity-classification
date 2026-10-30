from sklearn.utils import resample
import random
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, balanced_accuracy_score, confusion_matrix,
                             classification_report)
import torch
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = Path('C:\\Users\\montes\\OneDrive\\DPI\\2026-1\\paper Leonel\\Random_Forest_Info_Gain\\Random_Forest_Info_Gain\\github\\Data_IBARRAM.xlsx')
CHECKPOINT_PATH = Path('best_cnn_model.pth')
OUTPUT_CHECKPOINT = Path('best_cnn_model_finetuned80.pth')
SEED = 0
LEARNING_RATE = 0.001
MAX_EXTRA_EPOCHS = 7000

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def outlier_removal(feature, threshold):
    miu = feature.mean()
    sigma = feature.std()
    features_mantained = [i for i in range(len(feature)) if abs(feature[i] - miu) < (sigma * threshold)]
    return features_mantained


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
    return eye_matrix[np.asarray(y).astype(int) - 1]


def multiclass_specificity(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4])
    specificities = []
    for i in range(cm.shape[0]):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - TP - FP - FN
        specificities.append(TN / (TN + FP) if (TN + FP) > 0 else 0.0)
    return float(np.mean(specificities))

class CircleModelCNNV1(nn.Module):
    def __init__(self, layerdim, dropout_val=0.4, kernel_size=16):
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


def main():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f'File not found: {CHECKPOINT_PATH.resolve()}. '
            'Place best_cnn_model.pth for seed=0 in the working directory.'
        )
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f'Original dataset not found: {DATA_PATH}')


    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    if int(checkpoint.get('seed', -1)) != SEED:
        raise ValueError('The checkpoint does not certify seed=0; execution is cancelled to avoid using another model.')
    for field in ('state_dict', 'layerdim', 'kernel_size', 'dropout_val'):
        if field not in checkpoint:
            raise KeyError(f'Missing {field!r} in the checkpoint for CNN-1.')

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    model = CircleModelCNNV1(
        layerdim=int(checkpoint['layerdim']),
        dropout_val=float(checkpoint['dropout_val']),
        kernel_size=int(checkpoint['kernel_size'])
    ).to(DEVICE)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    loss_fn = nn.BCEWithLogitsLoss()


    data = pd.read_excel(DATA_PATH)
    data_int = convertObj_to_Int(data).dropna()
    X = data_int.drop('Fault', axis=1)
    y = data_int.Fault
    if set(y.unique()) != {1, 2, 3, 4}:
        raise ValueError('Exactly classes 1, 2, 3 and 4 were expected.')
    cols_A = [f'A{i}' for i in range(1, 501)]
    cols_B = [f'B{i}' for i in range(1, 501)]
    cols_C = [f'C{i}' for i in range(1, 501)]

    X_dev80, X_test20, y_dev80, y_test20 = train_test_split(
        X, y, test_size=0.2, stratify=y, shuffle=True, random_state=SEED
    )
    X_train64, X_old_val16, y_train64, y_old_val16 = train_test_split(
        X_dev80, y_dev80, test_size=0.2, stratify=y_dev80,
        shuffle=True, random_state=SEED
    )


    scaler_A, scaler_B, scaler_C = MinMaxScaler(), MinMaxScaler(), MinMaxScaler()
    scalers = [(scaler_A, cols_A), (scaler_B, cols_B), (scaler_C, cols_C)]
    for scaler, cols in scalers:
        scaler.fit(X_train64[cols].values)

    def transform_3phases(frame):
        return np.stack([scaler.transform(frame[cols].values)
                         for scaler, cols in scalers], axis=1).astype(np.float32)

    X64_normalized = transform_3phases(X_train64)
    X16_normalized = transform_3phases(X_old_val16)
    X64_filtered, y64_filtered = remove_training_outliers(X64_normalized, y_train64.values)
    X64_model, y64_model = balance_training_classes(X64_filtered, y64_filtered, random_state=SEED)
    X64_t = torch.from_numpy(X64_model).to(DEVICE)
    y64_t = torch.tensor(convert_to_onehot(y64_model), dtype=torch.float32, device=DEVICE)
    X16_t = torch.from_numpy(X16_normalized).to(DEVICE)
    y16_t = torch.tensor(convert_to_onehot(y_old_val16.values), dtype=torch.float32, device=DEVICE)
    X80_normalized = np.concatenate((X64_normalized, X16_normalized), axis=0)
    y80_raw = np.concatenate((y_train64.values, y_old_val16.values))
    X80_filtered, y80_filtered = remove_training_outliers(X80_normalized, y80_raw)
    X80_model, y80_model = balance_training_classes(X80_filtered, y80_filtered, random_state=SEED)
    X80_t = torch.from_numpy(X80_model).to(DEVICE)
    y80_t = torch.tensor(convert_to_onehot(y80_model), dtype=torch.float32, device=DEVICE)


    model.eval()
    with torch.inference_mode():
        target_loss = float(loss_fn(model(X64_t), y64_t).item())
        old_val_loss = float(loss_fn(model(X16_t), y16_t).item())

    print('\n' + '=' * 70)
    print('CNN-1: FINE-TUNING WITH 80% (SEED 0 REFERENCE CHECKPOINT)')
    print('=' * 70)
    print('Data: original 64% + former 16% validation = 80%')
    print(f'Original model: {CHECKPOINT_PATH}')
    print(f'Optimizer: new AdamW | constant LR: {LEARNING_RATE}')
    print(f'Loss threshold on 64%: {target_loss:.6f}')
    print(f'Initial loss on former 16%: {old_val_loss:.6f}')
    print(f'Maximum additional epochs: {MAX_EXTRA_EPOCHS}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    epochs_completed = 0
    history = [{'Extra_Epoch': 0, 'Train80_Loss': np.nan,
                'FormerVal16_Loss': old_val_loss, 'Target_Loss64': target_loss}]
    criterion_met = old_val_loss <= target_loss

    for epoch in range(1, MAX_EXTRA_EPOCHS + 1):
        if criterion_met:
            break
        model.train()
        optimizer.zero_grad()
        logits = model(X80_t)
        train_loss = loss_fn(logits, y80_t)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.inference_mode():
            old_val_loss = float(loss_fn(model(X16_t), y16_t).item())
        epochs_completed = epoch
        criterion_met = old_val_loss <= target_loss
        history.append({'Extra_Epoch': epoch, 'Train80_Loss': float(train_loss.item()),
                        'FormerVal16_Loss': old_val_loss, 'Target_Loss64': target_loss})
        print(f'Additional epoch {epoch:3d} | Loss on former 16%: {old_val_loss:.6f} '
              f'| Threshold: {target_loss:.6f}')

    status = ('criterion_met_without_updates' if epochs_completed == 0 else
              'criterion_reached' if criterion_met else 'epoch_limit_reached')
    if epochs_completed == 0:
        print('NOTICE: the criterion was initially satisfied; no updates were performed.')
    elif not criterion_met:
        print('NOTICE: the threshold was not reached within the preset limit.')


    new_checkpoint = {
        'state_dict': {k: v.detach().clone() for k, v in model.state_dict().items()},
        'seed': SEED,
        'layerdim': int(checkpoint['layerdim']),
        'kernel_size': int(checkpoint['kernel_size']),
        'dropout_val': float(checkpoint['dropout_val']),
        'source_checkpoint': str(CHECKPOINT_PATH),
        'training_fraction': 0.8 if epochs_completed > 0 else 0.64,
        'fixed_lr': float(LEARNING_RATE),
        'epochs_completed': int(epochs_completed),
        'target_loss_64': float(target_loss),
        'final_loss_former_16': float(old_val_loss),
        'criterion_met': bool(criterion_met),
        'status': status,
    }
    torch.save(new_checkpoint, OUTPUT_CHECKPOINT)
    pd.DataFrame(history).to_csv('CNN1_finetuning80_history.csv', index=False)
    print(f'\nContinuation checkpoint: {OUTPUT_CHECKPOINT} ({status})')


    Xtest_t = torch.from_numpy(transform_3phases(X_test20)).to(DEVICE)
    model.eval()
    with torch.inference_mode():
        logits80 = model(X80_t)
        logits_test = model(Xtest_t)
        train_pred = torch.argmax(torch.sigmoid(logits80), dim=1).cpu().numpy() + 1
        test_pred = torch.argmax(torch.sigmoid(logits_test), dim=1).cpu().numpy() + 1

        test_prob = torch.softmax(torch.sigmoid(logits_test), dim=1).cpu().numpy()

    yt = y_test20.values
    try:
        auc_test = float(roc_auc_score(yt, test_prob, average='weighted', multi_class='ovr'))
    except ValueError:
        auc_test = float('nan')
    result = {
        'Seed': SEED,
        'Extra_Epochs': epochs_completed,
        'Criterion_Met': criterion_met,
        'Stopping_Status': status,
        'Learning_Rate': LEARNING_RATE,
        'Target_Loss_64': target_loss,
        'Final_Loss_Former_16': old_val_loss,
        'Acc_Train80': accuracy_score(y80_t.argmax(dim=1).cpu().numpy() + 1, train_pred),
        'Acc_Test': accuracy_score(yt, test_pred),
        'Precision_Test': precision_score(yt, test_pred, average='weighted', zero_division=0),
        'Recall_Test': recall_score(yt, test_pred, average='weighted', zero_division=0),
        'F1_Test': f1_score(yt, test_pred, average='weighted', zero_division=0),
        'AUC_Test': auc_test,
        'Specificity_Test': multiclass_specificity(yt, test_pred),
        'Balanced_Acc_Test': balanced_accuracy_score(yt, test_pred),
    }
    pd.DataFrame([result]).to_csv('CNN1_finetuning80_test_results.csv', index=False)

    class_names = [
        '10% Pole (2.5% Phase)', '20% Pole (5.0% Phase)',
        '30% Pole (7.5% Phase)', '40% Pole (10.0% Phase)'
    ]
    report = classification_report(yt, test_pred, labels=[1, 2, 3, 4],
                                   target_names=class_names, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_excel('CNN1_finetuning80_classification_report.xlsx')
    cm = confusion_matrix(yt, test_pred, labels=[1, 2, 3, 4])
    plt.figure(figsize=(11, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('CNN-1 Fine-Tuning 80% - Test seed 0')
    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()
    plt.savefig('CNN1_finetuning80_confusion_matrix.png', dpi=300)

    plt.show()

    print('\n' + '=' * 70)
    print('FINAL EVALUATION: CNN-1 FINE-TUNING 80% - ORIGINAL TEST SEED 0')
    print('=' * 70)
    print(f'Additional epochs: {epochs_completed} | Status: {status}')
    print(f"Accuracy Test: {result['Acc_Test']:.4f}")
    print(f"Precision Test: {result['Precision_Test']:.4f}")
    print(f"Recall Test: {result['Recall_Test']:.4f}")
    print(f"F1-Score Test: {result['F1_Test']:.4f}")
    print(f"AUC Test: {result['AUC_Test']:.4f}")
    print(f"Balanced Acc Test: {result['Balanced_Acc_Test']:.4f}")
    print('=' * 70)


if __name__ == '__main__':
    main()
