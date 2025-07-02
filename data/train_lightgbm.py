import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from lightgbm import LGBMClassifier
import joblib

PROCESSED_DIR = './processed'

# Load data
X = pd.read_csv('processed/X_train.csv')
y = pd.read_csv('processed/y_train.csv')['Crop_encoded']

# Check class distribution
print("Original class distribution:")
class_counts = y.value_counts().sort_index()
print(class_counts)

# **SOLUTION 1: Remove classes with too few samples**
print("\nRemoving classes with less than 3 samples...")
min_samples = 3
valid_classes = class_counts[class_counts >= min_samples].index
print(f"Keeping {len(valid_classes)} classes out of {len(class_counts)}")

# Filter data to keep only valid classes
mask = y.isin(valid_classes)
X_filtered = X[mask]
y_filtered = y[mask]

print(f"\nFiltered data: {len(X_filtered)} samples, {len(y_filtered.unique())} classes")
print("New class distribution:")
print(y_filtered.value_counts().sort_index())

# **SOLUTION 2: Use simple train_test_split (not stratified)**
X_train, X_test, y_train, y_test = train_test_split(
    X_filtered, y_filtered, 
    test_size=0.2, 
    random_state=42,
    # Remove stratify parameter to avoid the error
    shuffle=True
)

print(f"\nTraining set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Train model with balanced class weights
model = LGBMClassifier(
    random_state=42,
    class_weight='balanced',  # This helps with remaining imbalance
    n_estimators=200,
    learning_rate=0.1,
    max_depth=8,
    verbose=-1
)

print("\nTraining model...")
model.fit(X_train, y_train)

# Evaluate model
from sklearn.metrics import accuracy_score, classification_report
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel accuracy: {accuracy:.4f}")

# Save the model
joblib.dump(model, 'processed/crop_recommendation_lgbm_balanced.pkl')
print("Balanced model saved!")

# **SOLUTION 3: Create a mapping for removed classes**
# Map removed classes to similar crops
class_mapping = {}
crop_label_df = pd.read_csv('processed/crop_label_map.csv', index_col=0)
crop_names = crop_label_df.to_dict()[0]  # Get crop names

removed_classes = set(class_counts.index) - set(valid_classes)
print(f"\nRemoved classes: {removed_classes}")
for removed_class in removed_classes:
    crop_name = crop_names.get(removed_class, f"Unknown_{removed_class}")
    print(f"Class {removed_class}: {crop_name}")

# Save class mapping info
mapping_info = {
    'valid_classes': valid_classes.tolist(),
    'removed_classes': list(removed_classes),
    'crop_names': crop_names
}

import json
with open('processed/class_mapping.json', 'w') as f:
    json.dump(mapping_info, f, indent=2)
print("Class mapping saved!")