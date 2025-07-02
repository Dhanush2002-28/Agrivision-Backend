import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
import joblib

# Load data
X = pd.read_csv('processed/X_train.csv')
y = pd.read_csv('processed/y_train.csv')['Crop_encoded']

print("Original class distribution:")
class_counts = y.value_counts().sort_index()
print(class_counts)

# **DATA AUGMENTATION: Add synthetic samples for rare classes**
min_samples = 5  # Target minimum samples per class
X_augmented = X.copy()
y_augmented = y.copy()

for class_label in class_counts.index:
    current_count = class_counts[class_label]
    if current_count < min_samples:
        # Get existing samples for this class
        class_mask = y == class_label
        class_samples = X[class_mask]
        
        # Generate synthetic samples by adding small noise
        needed_samples = min_samples - current_count
        print(f"Class {class_label}: Adding {needed_samples} synthetic samples")
        
        for _ in range(needed_samples):
            # Pick a random existing sample
            base_sample = class_samples.sample(n=1).iloc[0]
            
            # Add small random noise (±5% of the value)
            noise_factor = 0.05
            synthetic_sample = base_sample * (1 + np.random.uniform(-noise_factor, noise_factor, len(base_sample)))
            
            # Ensure reasonable bounds
            synthetic_sample = np.clip(synthetic_sample, base_sample * 0.8, base_sample * 1.2)
            
            # Add to dataset
            X_augmented = pd.concat([X_augmented, synthetic_sample.to_frame().T], ignore_index=True)
            y_augmented = pd.concat([y_augmented, pd.Series([class_label])], ignore_index=True)

print(f"\nAugmented dataset: {len(X_augmented)} samples")
print("New class distribution:")
print(y_augmented.value_counts().sort_index())

# Now train with stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X_augmented, y_augmented, 
    test_size=0.2, 
    stratify=y_augmented,  # Now this works!
    random_state=42
)

# Train model
model = LGBMClassifier(
    random_state=42,
    class_weight='balanced',
    n_estimators=200,
    learning_rate=0.1,
    max_depth=8,
    verbose=-1
)

print("Training augmented model...")
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'processed/crop_recommendation_lgbm_augmented.pkl')
print("Augmented model saved!")