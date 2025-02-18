import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE

# Set random seed for reproducibility
np.random.seed(42)

# Load the Breast Cancer dataset
print("Loading Breast Cancer dataset...")
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
feature_names = cancer.feature_names

print(f"Dataset shape: {X.shape}")
print(f"Number of features: {X.shape[1]}")
print(f"Number of samples: {X.shape[0]}")
print(f"Target distribution:\n{pd.Series(y).value_counts()}")
print(f"Classes: {cancer.target_names}")

# Convert to pandas DataFrame for easier manipulation
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df.drop('target', axis=1),
    df['target'],
    test_size=0.2,
    stratify=df['target'],
    random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---- BASELINE MODEL (BEFORE AUGMENTATION) ----
print("\n--- Training baseline model (before augmentation) ---")
baseline_model = MLPClassifier(hidden_layer_sizes=(64, 32),
                               max_iter=2000,
                               alpha=0.001,
                               activation='relu',
                               random_state=42)
baseline_model.fit(X_train_scaled, y_train)

# Evaluate the baseline model
y_pred_baseline = baseline_model.predict(X_test_scaled)
y_pred_proba_baseline = baseline_model.predict_proba(X_test_scaled)[:, 1]
baseline_accuracy = accuracy_score(y_test, y_pred_baseline)
baseline_cm = confusion_matrix(y_test, y_pred_baseline)

print(f"Baseline Model Accuracy: {baseline_accuracy:.4f}")
print("Baseline Classification Report:")
print(classification_report(y_test, y_pred_baseline, target_names=cancer.target_names))

# Calculate ROC curve for baseline model
fpr_baseline, tpr_baseline, _ = roc_curve(y_test, y_pred_proba_baseline)
roc_auc_baseline = auc(fpr_baseline, tpr_baseline)

# ---- DATA AUGMENTATION USING SMOTE ----
print("\n--- Applying data augmentation using SMOTE ---")
smote = SMOTE(random_state=42)
X_train_augmented, y_train_augmented = smote.fit_resample(X_train_scaled, y_train)

# Check new class distribution after SMOTE
augmented_counts = pd.Series(y_train_augmented).value_counts()
print(f"Class distribution after SMOTE:\n{augmented_counts}")

# Train model on augmented data
augmentation_model = MLPClassifier(hidden_layer_sizes=(64, 32),
                                   max_iter=2000,
                                   alpha=0.001,
                                   activation='relu',
                                   random_state=42)
augmentation_model.fit(X_train_augmented, y_train_augmented)

# Evaluate the augmentation model
y_pred_augmentation = augmentation_model.predict(X_test_scaled)
y_pred_proba_augmentation = augmentation_model.predict_proba(X_test_scaled)[:, 1]
augmentation_accuracy = accuracy_score(y_test, y_pred_augmentation)
augmentation_cm = confusion_matrix(y_test, y_pred_augmentation)

print(f"Augmentation Model Accuracy: {augmentation_accuracy:.4f}")
print("Augmentation Model Classification Report:")
print(classification_report(y_test, y_pred_augmentation, target_names=cancer.target_names))

# Calculate ROC curve for augmentation model
fpr_augmentation, tpr_augmentation, _ = roc_curve(y_test, y_pred_proba_augmentation)
roc_auc_augmentation = auc(fpr_augmentation, tpr_augmentation)

# ---- FEATURE ENGINEERING ----
print("\n--- Performing feature engineering ---")

# 1. Create a subset of important features to avoid excessive feature explosion
# Let's select top 15 features to keep feature engineering manageable
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=15)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

selected_features_idx = selector.get_support(indices=True)
selected_feature_names = [feature_names[i] for i in selected_features_idx]
print(f"Selected top features: {selected_feature_names}")

# 2. Polynomial features on the selected subset
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_selected)
X_test_poly = poly.transform(X_test_selected)

print(f"Original selected feature count: {X_train_selected.shape[1]}")
print(f"Feature count after polynomial transformation: {X_train_poly.shape[1]}")

# 3. Manual feature engineering on the selected features
X_train_eng_df = pd.DataFrame(X_train_selected, columns=selected_feature_names)
X_test_eng_df = pd.DataFrame(X_test_selected, columns=selected_feature_names)

# Create interaction and ratio features for selected pairs
for i, feat1 in enumerate(selected_feature_names):
    for j, feat2 in enumerate(selected_feature_names):
        if i < j:  # Avoid duplicate pairs and self-ratios
            # Product interactions
            X_train_eng_df[f'{feat1[:5]}_{feat2[:5]}_prod'] = X_train_eng_df[feat1] * X_train_eng_df[feat2]
            X_test_eng_df[f'{feat1[:5]}_{feat2[:5]}_prod'] = X_test_eng_df[feat1] * X_test_eng_df[feat2]

            # Ratio features with small epsilon to avoid division by zero
            X_train_eng_df[f'{feat1[:5]}_{feat2[:5]}_ratio'] = X_train_eng_df[feat1] / (X_train_eng_df[feat2] + 1e-10)
            X_test_eng_df[f'{feat1[:5]}_{feat2[:5]}_ratio'] = X_test_eng_df[feat1] / (X_test_eng_df[feat2] + 1e-10)

# Add some non-linear transformations of individual features
for feat in selected_feature_names:
    # Square
    X_train_eng_df[f'{feat[:5]}_squared'] = np.square(X_train_eng_df[feat])
    X_test_eng_df[f'{feat[:5]}_squared'] = np.square(X_test_eng_df[feat])

    # Square root (using absolute value to handle potentially negative scaled values)
    X_train_eng_df[f'{feat[:5]}_sqrt'] = np.sqrt(np.abs(X_train_eng_df[feat]))
    X_test_eng_df[f'{feat[:5]}_sqrt'] = np.sqrt(np.abs(X_test_eng_df[feat]))

    # Log transform (adding small epsilon and using absolute value)
    X_train_eng_df[f'{feat[:5]}_log'] = np.log(np.abs(X_train_eng_df[feat]) + 1e-10)
    X_test_eng_df[f'{feat[:5]}_log'] = np.log(np.abs(X_test_eng_df[feat]) + 1e-10)

print(f"Feature count after manual engineering: {X_train_eng_df.shape[1]}")

# Handle NaN or infinite values
X_train_eng_df.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test_eng_df.replace([np.inf, -np.inf], np.nan, inplace=True)
X_train_eng_df.fillna(0, inplace=True)
X_test_eng_df.fillna(0, inplace=True)

# Convert back to numpy arrays
X_train_eng = X_train_eng_df.values
X_test_eng = X_test_eng_df.values

# Train model on engineered features using augmented targets
feature_model = MLPClassifier(hidden_layer_sizes=(64, 32),
                              max_iter=2000,
                              alpha=0.001,
                              activation='relu',
                              random_state=42)
feature_model.fit(X_train_eng, y_train_augmented)

# Evaluate the feature engineering model
y_pred_feature = feature_model.predict(X_test_eng)
y_pred_proba_feature = feature_model.predict_proba(X_test_eng)[:, 1]
feature_accuracy = accuracy_score(y_test, y_pred_feature)
feature_cm = confusion_matrix(y_test, y_pred_feature)

print(f"Feature Engineering Model Accuracy: {feature_accuracy:.4f}")
print("Feature Engineering Model Classification Report:")
print(classification_report(y_test, y_pred_feature, target_names=cancer.target_names))

# Calculate ROC curve for feature engineering model
fpr_feature, tpr_feature, _ = roc_curve(y_test, y_pred_proba_feature)
roc_auc_feature = auc(fpr_feature, tpr_feature)

# ---- PLOT CONFUSION MATRICES ----
plt.figure(figsize=(18, 6))

# Baseline model confusion matrix
plt.subplot(1, 3, 1)
sns.heatmap(baseline_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=cancer.target_names,
            yticklabels=cancer.target_names)
plt.title(f'Baseline Model\nAccuracy: {baseline_accuracy:.4f}')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Augmentation model confusion matrix
plt.subplot(1, 3, 2)
sns.heatmap(augmentation_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=cancer.target_names,
            yticklabels=cancer.target_names)
plt.title(f'After Augmentation\nAccuracy: {augmentation_accuracy:.4f}')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Feature engineering model confusion matrix
plt.subplot(1, 3, 3)
sns.heatmap(feature_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=cancer.target_names,
            yticklabels=cancer.target_names)
plt.title(f'After Feature Engineering\nAccuracy: {feature_accuracy:.4f}')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.tight_layout()
plt.show()

# ---- PLOT ROC CURVES ----
plt.figure(figsize=(10, 8))
plt.plot(fpr_baseline, tpr_baseline, color='blue',
         label=f'Baseline (AUC = {roc_auc_baseline:.4f})')
plt.plot(fpr_augmentation, tpr_augmentation, color='green',
         label=f'After Augmentation (AUC = {roc_auc_augmentation:.4f})')
plt.plot(fpr_feature, tpr_feature, color='red',
         label=f'After Feature Engineering (AUC = {roc_auc_feature:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc="lower right")
plt.show()

# Compare accuracies
methods = ['Baseline', 'After Augmentation', 'After Feature Engineering']
accuracies = [baseline_accuracy, augmentation_accuracy, feature_accuracy]

plt.figure(figsize=(10, 6))
ax = sns.barplot(x=methods, y=accuracies)
plt.title('Comparison of Model Accuracies')
plt.ylabel('Accuracy')
plt.ylim(0.9, 1.0)  # Adjust for better visibility of differences

# Add text annotations on bars
for i, acc in enumerate(accuracies):
    ax.text(i, acc + 0.005, f'{acc:.4f}', ha='center')

plt.tight_layout()
plt.show()

# Print summary of improvements
print("\n--- SUMMARY OF IMPROVEMENTS ---")
print(f"Baseline accuracy: {baseline_accuracy:.4f}")
print(
    f"Accuracy after augmentation: {augmentation_accuracy:.4f} (Change: {augmentation_accuracy - baseline_accuracy:.4f})")
print(
    f"Accuracy after feature engineering: {feature_accuracy:.4f} (Change from baseline: {feature_accuracy - baseline_accuracy:.4f})")
print(f"Total improvement: {feature_accuracy - baseline_accuracy:.4f}")

# Compare AUC values
print("\n--- AUC COMPARISON ---")
print(f"Baseline AUC: {roc_auc_baseline:.4f}")
print(f"AUC after augmentation: {roc_auc_augmentation:.4f} (Change: {roc_auc_augmentation - roc_auc_baseline:.4f})")
print(
    f"AUC after feature engineering: {roc_auc_feature:.4f} (Change from baseline: {roc_auc_feature - roc_auc_baseline:.4f})")
print(f"Total AUC improvement: {roc_auc_feature - roc_auc_baseline:.4f}")