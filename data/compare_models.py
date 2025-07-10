import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                           precision_score, recall_score, f1_score, roc_auc_score,
                           log_loss, matthews_corrcoef, balanced_accuracy_score)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.multiclass import OneVsRestClassifier
import joblib
import time
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import all 5 classification models
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# Also try XGBoost as bonus
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not installed. Install with: pip install xgboost")

def load_and_prepare_data():
    """Load and prepare the crop recommendation data"""
    print("Loading data...")
    
    # Load data
    X = pd.read_csv('processed/X_train.csv')
    y = pd.read_csv('processed/y_train.csv')['Crop_encoded']
    
    print(f"Original dataset: {len(X)} samples, {len(y.unique())} classes")
    print(f"Original class labels: {sorted(y.unique())}")
    
    # Handle class imbalance by removing classes with too few samples
    class_counts = y.value_counts()
    min_samples = 3
    valid_classes = class_counts[class_counts >= min_samples].index
    
    mask = y.isin(valid_classes)
    X_filtered = X[mask]
    y_filtered = y[mask]
    
    print(f"Filtered dataset: {len(X_filtered)} samples, {len(y_filtered.unique())} classes")
    print(f"Filtered class labels: {sorted(y_filtered.unique())}")
    
    # Remap class labels to be contiguous (0, 1, 2, ..., n-1)
    # This is required for XGBoost and some other algorithms
    unique_labels = sorted(y_filtered.unique())
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    
    # Apply the mapping
    y_remapped = y_filtered.map(label_mapping)
    
    print(f"Remapped class labels: {sorted(y_remapped.unique())}")
    print(f"Label mapping: {label_mapping}")
    
    # Save label mapping for later use (if needed for predictions)
    label_mapping_df = pd.DataFrame(list(label_mapping.items()), 
                                   columns=['original_label', 'new_label'])
    label_mapping_df.to_csv('processed/label_mapping.csv', index=False)
    print("Label mapping saved to processed/label_mapping.csv")
    
    return X_filtered, y_remapped

def create_models():
    """Create all 5 classification models"""
    models = {}
    
    # 1. Random Forest
    models['Random Forest'] = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    
    # 2. LightGBM (your current model)
    models['LightGBM'] = LGBMClassifier(
        random_state=42,
        class_weight='balanced',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=8,
        verbose=-1
    )
    
    # 3. Support Vector Machine
    models['SVM'] = SVC(
        kernel='rbf',
        probability=True,  # Enable probability estimates
        random_state=42,
        class_weight='balanced'
    )
    
    # 4. Logistic Regression
    models['Logistic Regression'] = LogisticRegression(
        random_state=42,
        class_weight='balanced',
        multi_class='ovr',  # One-vs-Rest for multi-class
        max_iter=1000
    )
    
    # 5. Neural Network (MLP)
    models['Neural Network'] = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        random_state=42,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1
    )
    
    # Bonus: XGBoost (if available)
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = XGBClassifier(
            random_state=42,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=8,
            eval_metric='mlogloss'
        )
    
    return models

def calculate_model_specific_metrics(model_name, model, X_test, y_test, y_pred, y_proba=None):
    """
    Calculate metrics that are most important for each specific algorithm
    """
    metrics = {}
    
    # Common metrics for all models
    metrics['accuracy'] = accuracy_score(y_test, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_test, y_pred)
    metrics['f1_macro'] = f1_score(y_test, y_pred, average='macro')
    metrics['f1_weighted'] = f1_score(y_test, y_pred, average='weighted')
    
    # Model-specific key metrics
    if model_name == 'Random Forest':
        # Random Forest: Focus on OOB score, feature importance, generalization
        try:
            if hasattr(model, 'oob_score_') and model.oob_score_ is not None:
                metrics['oob_score'] = model.oob_score_
            else:
                metrics['oob_score'] = None
            metrics['feature_importance_std'] = np.std(model.feature_importances_)
            metrics['key_metric'] = 'oob_score'
            metrics['key_metric_name'] = 'Out-of-Bag Score'
        except:
            metrics['oob_score'] = None
            metrics['key_metric'] = 'f1_macro'
            metrics['key_metric_name'] = 'F1-Score (Macro)'
    
    elif model_name in ['LightGBM', 'XGBoost']:
        # Gradient Boosting: Focus on log loss, AUC, early stopping validation
        if y_proba is not None:
            try:
                metrics['log_loss'] = log_loss(y_test, y_proba)
                # Multi-class AUC (One-vs-Rest)
                y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
                if y_test_bin.shape[1] > 1:
                    metrics['auc_ovr'] = roc_auc_score(y_test_bin, y_proba, multi_class='ovr')
                else:
                    metrics['auc_ovr'] = None
                metrics['key_metric'] = 'log_loss'
                metrics['key_metric_name'] = 'Log Loss (lower is better)'
            except:
                metrics['log_loss'] = None
                metrics['auc_ovr'] = None
                metrics['key_metric'] = 'f1_weighted'
                metrics['key_metric_name'] = 'F1-Score (Weighted)'
        else:
            metrics['key_metric'] = 'f1_weighted'
            metrics['key_metric_name'] = 'F1-Score (Weighted)'
    
    elif model_name == 'SVM':
        # SVM: Focus on margin-based metrics, support vector count
        try:
            metrics['n_support_vectors'] = np.sum(model.n_support_)
            metrics['support_vector_ratio'] = metrics['n_support_vectors'] / len(X_test)
            # Hinge loss approximation using decision function
            decision_scores = model.decision_function(X_test)
            metrics['decision_score_std'] = np.std(decision_scores)
            metrics['key_metric'] = 'balanced_accuracy'
            metrics['key_metric_name'] = 'Balanced Accuracy'
        except:
            metrics['key_metric'] = 'balanced_accuracy'
            metrics['key_metric_name'] = 'Balanced Accuracy'
    
    elif model_name == 'Logistic Regression':
        # Logistic Regression: Focus on log loss, calibration
        if y_proba is not None:
            try:
                metrics['log_loss'] = log_loss(y_test, y_proba)
                # Calibration - Brier score approximation
                metrics['avg_predicted_prob'] = np.mean(np.max(y_proba, axis=1))
                metrics['key_metric'] = 'log_loss'
                metrics['key_metric_name'] = 'Log Loss (lower is better)'
            except:
                metrics['log_loss'] = None
                metrics['key_metric'] = 'f1_macro'
                metrics['key_metric_name'] = 'F1-Score (Macro)'
        else:
            metrics['key_metric'] = 'f1_macro'
            metrics['key_metric_name'] = 'F1-Score (Macro)'
    
    elif model_name == 'Neural Network':
        # Neural Network: Focus on loss convergence, overfitting
        try:
            metrics['n_iterations'] = model.n_iter_
            if hasattr(model, 'loss_curve_'):
                metrics['final_loss'] = model.loss_curve_[-1]
                if len(model.loss_curve_) > 10:
                    metrics['loss_improvement'] = model.loss_curve_[10] - model.loss_curve_[-1]
                else:
                    metrics['loss_improvement'] = None
            if y_proba is not None:
                metrics['log_loss'] = log_loss(y_test, y_proba)
                metrics['key_metric'] = 'log_loss'
                metrics['key_metric_name'] = 'Log Loss (lower is better)'
            else:
                metrics['key_metric'] = 'f1_weighted'
                metrics['key_metric_name'] = 'F1-Score (Weighted)'
        except:
            metrics['key_metric'] = 'f1_weighted'
            metrics['key_metric_name'] = 'F1-Score (Weighted)'
    
    else:
        # Default metrics
        metrics['key_metric'] = 'f1_macro'
        metrics['key_metric_name'] = 'F1-Score (Macro)'
    
    # Matthews Correlation Coefficient (good for imbalanced datasets)
    try:
        metrics['mcc'] = matthews_corrcoef(y_test, y_pred)
    except:
        metrics['mcc'] = None
    
    return metrics

def evaluate_model(model, model_name, X_train, X_test, y_train, y_test, scaler=None):
    """Train and evaluate a single model with algorithm-specific metrics"""
    print(f"\n{'='*60}")
    print(f"🔍 Training {model_name}...")
    print(f"{'='*60}")
    
    # Special handling for Random Forest to enable OOB score
    if model_name == 'Random Forest':
        model.set_params(oob_score=True)
    
    # Scale data for models that need it
    if scaler and model_name in ['SVM', 'Neural Network', 'Logistic Regression']:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        print(f"📊 Data scaled for {model_name}")
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
    
    # Training
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time
    
    # Prediction
    start_time = time.time()
    y_pred = model.predict(X_test_scaled)
    prediction_time = time.time() - start_time
    
    # Probability prediction (if available)
    y_proba = None
    if hasattr(model, 'predict_proba'):
        try:
            y_proba = model.predict_proba(X_test_scaled)
        except:
            pass
    
    # Calculate model-specific metrics
    metrics = calculate_model_specific_metrics(model_name, model, X_test_scaled, y_test, y_pred, y_proba)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    
    # Compile results
    results = {
        'model_name': model_name,
        'training_time': training_time,
        'prediction_time': prediction_time,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'y_pred': y_pred,
        **metrics  # Include all calculated metrics
    }
    
    # Print results
    print(f"📈 Accuracy: {metrics['accuracy']:.4f}")
    print(f"📊 Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"🎯 F1-Score (Macro): {metrics['f1_macro']:.4f}")
    print(f"⚖️ F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
    if metrics.get('mcc') is not None:
        print(f"🔗 Matthews Correlation: {metrics['mcc']:.4f}")
    print(f"🔄 CV Score: {cv_scores.mean():.4f} (±{cv_scores.std() * 2:.4f})")
    print(f"⏱️ Training Time: {training_time:.2f}s")
    print(f"⚡ Prediction Time: {prediction_time:.4f}s")
    
    # Print algorithm-specific key metric
    key_metric_value = metrics.get(metrics['key_metric'])
    if key_metric_value is not None:
        print(f"🌟 {metrics['key_metric_name']}: {key_metric_value:.4f}")
    
    # Print algorithm-specific insights
    if model_name == 'Random Forest' and metrics.get('oob_score') is not None:
        print(f"🌳 Out-of-Bag Score: {metrics['oob_score']:.4f}")
        print(f"📊 Feature Importance Std: {metrics.get('feature_importance_std', 0):.4f}")
    elif model_name in ['LightGBM', 'XGBoost'] and metrics.get('log_loss') is not None:
        print(f"📉 Log Loss: {metrics['log_loss']:.4f}")
        if metrics.get('auc_ovr') is not None:
            print(f"📈 AUC (OvR): {metrics['auc_ovr']:.4f}")
    elif model_name == 'SVM' and metrics.get('n_support_vectors') is not None:
        print(f"🎯 Support Vectors: {metrics['n_support_vectors']}")
        print(f"📊 SV Ratio: {metrics['support_vector_ratio']:.4f}")
    elif model_name == 'Neural Network' and metrics.get('n_iterations') is not None:
        print(f"🔄 Iterations: {metrics['n_iterations']}")
        if metrics.get('final_loss') is not None:
            print(f"📉 Final Loss: {metrics['final_loss']:.4f}")
    
    return results, model

def plot_results(results_df):
    """Create comprehensive visualizations comparing all models"""
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    
    # 1. Accuracy Comparison
    bars1 = axes[0, 0].bar(results_df['model_name'], results_df['accuracy'], 
                          color='skyblue', alpha=0.8)
    axes[0, 0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].set_ylim(0, 1)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom')
    
    # 2. F1-Score Comparison (Macro and Weighted)
    x_pos = np.arange(len(results_df))
    width = 0.35
    axes[0, 1].bar(x_pos - width/2, results_df['f1_macro'], width, 
                   label='F1-Macro', alpha=0.8, color='lightcoral')
    axes[0, 1].bar(x_pos + width/2, results_df['f1_weighted'], width,
                   label='F1-Weighted', alpha=0.8, color='lightgreen')
    axes[0, 1].set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('F1-Score')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(results_df['model_name'], rotation=45)
    axes[0, 1].legend()
    axes[0, 1].set_ylim(0, 1)
    
    # 3. Cross-Validation Scores with Error Bars
    axes[1, 0].bar(results_df['model_name'], results_df['cv_mean'], 
                   yerr=results_df['cv_std'], capsize=5, color='lightblue', alpha=0.8)
    axes[1, 0].set_title('Cross-Validation Scores (±1 std)', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('CV Score')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].set_ylim(0, 1)
    
    # 4. Training Time Comparison
    bars4 = axes[1, 1].bar(results_df['model_name'], results_df['training_time'], 
                          color='orange', alpha=0.8)
    axes[1, 1].set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Time (seconds)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar in bars4:
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{height:.2f}s', ha='center', va='bottom')
    
    # 5. Speed vs Accuracy Trade-off
    scatter = axes[2, 0].scatter(results_df['training_time'], results_df['accuracy'], 
                                s=200, alpha=0.7, c=range(len(results_df)), cmap='viridis')
    for i, txt in enumerate(results_df['model_name']):
        axes[2, 0].annotate(txt, (results_df['training_time'].iloc[i], 
                                 results_df['accuracy'].iloc[i]),
                           xytext=(5, 5), textcoords='offset points')
    axes[2, 0].set_xlabel('Training Time (seconds)')
    axes[2, 0].set_ylabel('Accuracy')
    axes[2, 0].set_title('Speed vs Accuracy Trade-off', fontsize=14, fontweight='bold')
    axes[2, 0].grid(True, alpha=0.3)
    
    # 6. Algorithm-Specific Key Metrics Radar Chart
    # Create a comparison of balanced accuracy, F1-macro, and MCC
    metrics_to_compare = ['balanced_accuracy', 'f1_macro']
    if 'mcc' in results_df.columns and not results_df['mcc'].isna().all():
        metrics_to_compare.append('mcc')
    
    x_pos = np.arange(len(results_df))
    bar_width = 0.25
    
    for i, metric in enumerate(metrics_to_compare):
        if metric in results_df.columns:
            axes[2, 1].bar(x_pos + i * bar_width, results_df[metric], 
                          bar_width, label=metric.replace('_', ' ').title(), alpha=0.8)
    
    axes[2, 1].set_title('Key Metrics Comparison', fontsize=14, fontweight='bold')
    axes[2, 1].set_ylabel('Score')
    axes[2, 1].set_xticks(x_pos + bar_width)
    axes[2, 1].set_xticklabels(results_df['model_name'], rotation=45)
    axes[2, 1].legend()
    axes[2, 1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('processed/comprehensive_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_detailed_comparison_table(results_df):
    """Create a detailed comparison table with algorithm-specific metrics"""
    print(f"\n{'='*120}")
    print("📊 COMPREHENSIVE MODEL COMPARISON - ALGORITHM-SPECIFIC METRICS")
    print(f"{'='*120}")
    
    # Define columns to display
    display_columns = [
        'model_name', 'accuracy', 'balanced_accuracy', 'f1_macro', 'f1_weighted', 
        'cv_mean', 'training_time', 'prediction_time'
    ]
    
    # Add algorithm-specific metrics
    specific_metrics = {
        'Random Forest': ['oob_score'],
        'LightGBM': ['log_loss', 'auc_ovr'],
        'XGBoost': ['log_loss', 'auc_ovr'],
        'SVM': ['n_support_vectors', 'support_vector_ratio'],
        'Logistic Regression': ['log_loss'],
        'Neural Network': ['n_iterations', 'final_loss']
    }
    
    # Print header
    print(f"{'Model':<18} {'Accuracy':<9} {'Bal.Acc':<8} {'F1-Mac':<7} {'F1-Wei':<7} "
          f"{'CV±Std':<10} {'Train(s)':<8} {'Pred(s)':<7} {'Key Metric':<15}")
    print("-" * 120)
    
    # Print each model's results
    for _, row in results_df.iterrows():
        model_name = row['model_name']
        
        # Format basic metrics
        line = f"{model_name:<18} {row['accuracy']:<9.4f} {row['balanced_accuracy']:<8.4f} "
        line += f"{row['f1_macro']:<7.4f} {row['f1_weighted']:<7.4f} "
        line += f"{row['cv_mean']:.3f}±{row['cv_std']:.3f}  {row['training_time']:<8.2f} "
        line += f"{row['prediction_time']:<7.4f} "
        
        # Add key metric
        key_metric = row.get('key_metric')
        key_value = row.get(key_metric) if key_metric else None
        if key_value is not None:
            if key_metric in ['log_loss']:
                line += f"{key_value:.4f} (loss)"
            else:
                line += f"{key_value:.4f}"
        else:
            line += "N/A"
        
        print(line)
        
        # Print algorithm-specific details
        if model_name in specific_metrics:
            details = []
            for metric in specific_metrics[model_name]:
                value = row.get(metric)
                if value is not None:
                    if metric == 'n_support_vectors':
                        details.append(f"SV: {int(value)}")
                    elif metric == 'support_vector_ratio':
                        details.append(f"SV Ratio: {value:.3f}")
                    elif metric == 'oob_score':
                        details.append(f"OOB: {value:.4f}")
                    elif metric == 'log_loss':
                        details.append(f"LogLoss: {value:.4f}")
                    elif metric == 'auc_ovr':
                        details.append(f"AUC: {value:.4f}")
                    elif metric == 'n_iterations':
                        details.append(f"Iter: {int(value)}")
                    elif metric == 'final_loss':
                        details.append(f"Loss: {value:.4f}")
            
            if details:
                print(f"{'':18} {' | '.join(details)}")
        
        print()
    
    return results_df

def main():
    """Main function to run the complete model comparison"""
    print("🚀 Starting Comprehensive Model Comparison for Crop Recommendation")
    print("=" * 70)
    
    # Load data
    X, y = load_and_prepare_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Create scaler for models that need it
    scaler = StandardScaler()
    
    # Create all models
    models = create_models()
    print(f"\nWill test {len(models)} models: {list(models.keys())}")
    
    # Store results
    all_results = []
    trained_models = {}
    
    # Train and evaluate each model
    for model_name, model in models.items():
        try:
            results, trained_model = evaluate_model(
                model, model_name, X_train, X_test, y_train, y_test, scaler
            )
            all_results.append(results)
            trained_models[model_name] = trained_model
            
        except Exception as e:
            print(f"❌ Error training {model_name}: {str(e)}")
            continue
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Sort by accuracy
    results_df = results_df.sort_values('accuracy', ascending=False)
    
    # Print final comparison
    print(f"\n{'='*70}")
    print("🏆 FINAL MODEL COMPARISON RESULTS")
    print(f"{'='*70}")
    print(f"{'Model':<18} {'Accuracy':<10} {'CV Score':<12} {'Train Time':<12} {'Pred Time':<10}")
    print("-" * 70)
    
    for _, row in results_df.iterrows():
        print(f"{row['model_name']:<18} {row['accuracy']:<10.4f} "
              f"{row['cv_mean']:<12.4f} {row['training_time']:<12.2f} "
              f"{row['prediction_time']:<10.4f}")
    
    # Save results
    results_df.to_csv('processed/model_comparison_results.csv', index=False)
    
    # Save the best model
    best_model_name = results_df.iloc[0]['model_name']
    best_model = trained_models[best_model_name]
    joblib.dump(best_model, f'processed/best_model_{best_model_name.lower().replace(" ", "_")}.pkl')
    
    print(f"\n🎯 Best Model: {best_model_name} with accuracy: {results_df.iloc[0]['accuracy']:.4f}")
    print(f"📊 Results saved to: processed/model_comparison_results.csv")
    print(f"💾 Best model saved to: processed/best_model_{best_model_name.lower().replace(' ', '_')}.pkl")
    
    # Create visualizations
    try:
        plot_results(results_df)
        print(f"📈 Visualizations saved to: processed/model_comparison.png")
    except Exception as e:
        print(f"⚠️ Could not create plots: {str(e)}")
    
    # Detailed analysis for top 3 models
    print(f"\n{'='*70}")
    print("📋 DETAILED ANALYSIS - TOP 3 MODELS")
    print(f"{'='*70}")
    
    for i, (_, row) in enumerate(results_df.head(3).iterrows()):
        print(f"\n{i+1}. {row['model_name']}:")
        print(f"   • Accuracy: {row['accuracy']:.4f}")
        print(f"   • Cross-validation: {row['cv_mean']:.4f} ± {row['cv_std']:.4f}")
        print(f"   • Training time: {row['training_time']:.2f}s")
        print(f"   • Prediction time: {row['prediction_time']:.4f}s")
    
    print(f"\n🎉 Model comparison complete!")
    
    return results_df, trained_models

if __name__ == "__main__":
    results_df, trained_models = main()
