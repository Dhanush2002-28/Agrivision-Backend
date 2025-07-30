"""
SOLUTION 5 - SIMPLIFIED: Categorical Crop Recommendation System
Convert numerical crop data to categorical, merge with soil data, train ML models
"""
import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. LOAD DATASETS
# =============================================================================
def set_global_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    try:
        import os
        os.environ["PYTHONHASHSEED"] = str(seed)
    except Exception:
        pass

def load_datasets():
    """Load crop, soil, and weather datasets"""
    df_crop = pd.read_csv(r'C:\Users\DHANUSH C\OneDrive\Desktop\AgriVision_Project\Project\Agrivision_Backend\POV_Crops.csv')
    df_soil = pd.read_csv(r'C:\Users\DHANUSH C\OneDrive\Desktop\AgriVision_Project\Project\Agrivision_Backend\SoilData_DominantCategories.csv')
    df_weather = pd.read_csv(r'C:\Users\DHANUSH C\OneDrive\Desktop\AgriVision_Project\Project\Agrivision_Backend\Weather Data.csv')
    
    print("Dataset Loaded:")
    print(f"- Crop dataset: {df_crop.shape}")
    print(f"- Soil dataset: {df_soil.shape}")
    print(f"- Weather dataset: {df_weather.shape}")
    
    return df_crop, df_soil, df_weather

# =============================================================================
# 2. CONVERT NUMERICAL TO CATEGORICAL
# =============================================================================
def convert_to_categorical(df_crop):
    """Convert numerical NPK and environmental values to Low/Medium/High categories"""
    df_cat = df_crop.copy()
    
    df_cat['Nitrogen_Status'] = pd.cut(df_crop['Nitrogen'], 
                                      bins=[0, df_crop['Nitrogen'].quantile(0.33), df_crop['Nitrogen'].quantile(0.67), float('inf')],
                                      labels=['Low', 'Medium', 'High'])
    
    df_cat['Phosphorous_Status'] = pd.cut(df_crop['Phosphorus'], 
                                         bins=[0, df_crop['Phosphorus'].quantile(0.33), df_crop['Phosphorus'].quantile(0.67), float('inf')],
                                         labels=['Low', 'Medium', 'High'])
    
    df_cat['Potassium_Status'] = pd.cut(df_crop['Potassium'], 
                                       bins=[0, df_crop['Potassium'].quantile(0.33), df_crop['Potassium'].quantile(0.67), float('inf')],
                                       labels=['Low', 'Medium', 'High'])
    
    # pH: Acidic < 6.5, Neutral 6.5-7.5, Alkaline > 7.5
    df_cat['pH_Status'] = pd.cut(df_crop['pH_Value'], 
                                bins=[0, 6.5, 7.5, float('inf')],
                                labels=['Acidic', 'Neutral', 'Alkaline'])
    
    # Environmental factors (Temperature, Humidity, Rainfall) using quantiles
    df_cat['Temperature_Status_crop'] = pd.cut(df_crop['Temperature'], 
                                         bins=[0, df_crop['Temperature'].quantile(0.33), df_crop['Temperature'].quantile(0.67), float('inf')],
                                         labels=['Low', 'Medium', 'High'])
    
    df_cat['Humidity_Status_crop'] = pd.cut(df_crop['Humidity'], 
                                      bins=[0, df_crop['Humidity'].quantile(0.33), df_crop['Humidity'].quantile(0.67), float('inf')],
                                      labels=['Low', 'Medium', 'High'])
    
    df_cat['Rainfall_Status_crop'] = pd.cut(df_crop['Rainfall'], 
                                      bins=[0, df_crop['Rainfall'].quantile(0.33), df_crop['Rainfall'].quantile(0.67), float('inf')],
                                      labels=['Low', 'Medium', 'High'])
    
    print("✅ Converted numerical features to categorical (Low/Medium/High)")
    return df_cat

# =============================================================================
# 3. PREPARE WEATHER DATA
# =============================================================================
def prepare_weather_data(df_weather):
    """Convert weather data to categorical and get latest data for each state"""
    # Get latest year data for each state
    df_weather_latest = df_weather.loc[df_weather.groupby('State')['year'].idxmax()].copy()
    
    # Convert temperature to categorical using quantiles
    df_weather_latest['Temperature_Status_weather'] = pd.cut(df_weather_latest['Avg_Temp_C'], 
                                                    bins=[0, df_weather_latest['Avg_Temp_C'].quantile(0.33), 
                                                         df_weather_latest['Avg_Temp_C'].quantile(0.67), float('inf')],
                                                    labels=['Low', 'Medium', 'High'])
    
    # Convert rainfall to categorical using quantiles  
    df_weather_latest['Rainfall_Status_weather'] = pd.cut(df_weather_latest['Total_Rainfall_mm'], 
                                                  bins=[0, df_weather_latest['Total_Rainfall_mm'].quantile(0.33), 
                                                       df_weather_latest['Total_Rainfall_mm'].quantile(0.67), float('inf')],
                                                  labels=['Low', 'Medium', 'High'])
    
    # For humidity, we'll use a reasonable assumption based on rainfall and temperature
    # High rainfall + Medium/Low temp = High humidity
    # Low rainfall + High temp = Low humidity
    # Everything else = Medium humidity
    conditions = [
        (df_weather_latest['Rainfall_Status_weather'] == 'High') & (df_weather_latest['Temperature_Status_weather'].isin(['Low', 'Medium'])),
        (df_weather_latest['Rainfall_Status_weather'] == 'Low') & (df_weather_latest['Temperature_Status_weather'] == 'High')
    ]
    choices = ['High', 'Low']
    df_weather_latest['Humidity_Status_weather'] = np.select(conditions, choices, default='Medium')
    
    print(f"✅ Processed weather data for {len(df_weather_latest)} states")
    print("Weather status distribution:")
    print(f"Temperature: {df_weather_latest['Temperature_Status_weather'].value_counts().to_dict()}")
    print(f"Rainfall: {df_weather_latest['Rainfall_Status_weather'].value_counts().to_dict()}")
    print(f"Humidity: {df_weather_latest['Humidity_Status_weather'].value_counts().to_dict()}")
    
    return df_weather_latest[['State', 'Temperature_Status_weather', 'Rainfall_Status_weather', 'Humidity_Status_weather']]

# =============================================================================
# 4. MERGE CROP AND SOIL DATA
# =============================================================================
def merge_crop_soil_data(df_crop_cat, df_soil, df_weather_processed):
    """Assign random locations to crop data and merge with soil and weather data"""
    # Get unique locations from soil data
    soil_locations = df_soil[['State', 'District', 'Block']].drop_duplicates()

    # Assign random locations to crop data for merging
    crop_with_location = df_crop_cat.copy()
    random_locations = soil_locations.sample(n=len(crop_with_location), replace=True, random_state=42).reset_index(drop=True)
    crop_with_location[['State', 'District', 'Block']] = random_locations[['State', 'District', 'Block']]

    # Merge crop data with soil data based on location
    merged_data = crop_with_location.merge(df_soil, on=['State', 'District', 'Block'], how='left', suffixes=('_crop', '_soil'))

    # Merge with weather data based on state
    merged_data = merged_data.merge(df_weather_processed, on='State', how='left', suffixes=('', '_weather'))

    print(f"✅ Merged crop, soil, and weather data: {merged_data.shape}")
    print(f"✅ Successful soil merges: {merged_data['Nitrogen_Status_soil'].notna().sum()}/{len(merged_data)}")
    print(f"✅ Successful weather merges: {merged_data['Temperature_Status_weather'].notna().sum()}/{len(merged_data)}")

    return merged_data

# =============================================================================
# 5. PREPARE DATA FOR MACHINE LEARNING
# =============================================================================
def prepare_ml_data(merged_data):
    """Prepare categorical features for machine learning"""
    # Define categorical features from crop, soil, and weather data
    categorical_features = [
        'Nitrogen_Status_crop', 'Phosphorous_Status_crop', 'Potassium_Status_crop', 'pH_Status_crop',
        'Temperature_Status_crop', 'Humidity_Status_crop', 'Rainfall_Status_crop',  # From crop data conversion
        'Temperature_Status_weather', 'Humidity_Status_weather', 'Rainfall_Status_weather',  # From weather data
        'Nitrogen_Status_soil', 'Phosphorous_Status_soil', 'Potassium_Status_soil', 'pH_Status_soil',
        'OC_Status', 'EC_Status'
    ]
    
    # Get clean data without missing soil information
    merged_clean = merged_data.dropna(subset=['Nitrogen_Status_soil'])
    
    # Extract features and target
    feature_data = merged_clean[categorical_features].copy()
    
    # Convert categorical columns to string to avoid pandas categorical issues
    for col in categorical_features:
        if col in feature_data.columns:
            feature_data[col] = feature_data[col].astype(str)
    
    # Fill any missing values
    feature_data = feature_data.fillna('Unknown')
    target = merged_clean['Crop']
    
    # One-hot encode categorical features
    X_encoded = pd.get_dummies(feature_data, drop_first=False)
    y = target
    
    print(f"✅ Prepared ML data: {X_encoded.shape[0]} samples, {X_encoded.shape[1]} features, {y.nunique()} crops")
    
    return X_encoded, y, merged_clean

# =============================================================================
# 6. TRAIN MACHINE LEARNING MODELS
# =============================================================================
def train_models(X, y):
    """Train multiple ML models and return the best one"""
    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Define models to compare
    models = [
        ("RandomForest", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("LogisticRegression", LogisticRegression(max_iter=1000, random_state=42, multi_class='ovr')),
        ("SVM", SVC(probability=True, random_state=42))
    ]
    
    results = []
    trained_models = {}
    
    print("🤖 Training Models:")
    print("-" * 40)
    
    # Train and evaluate each model
    for name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"{name}: {acc:.4f} accuracy")
        results.append((name, acc))
        trained_models[name] = model
    
    # Find best model
    best_model = max(results, key=lambda x: x[1])
    print(f"\n🏆 Best Model: {best_model[0]} with {best_model[1]:.4f} accuracy")
    
    return trained_models, best_model, X

# =============================================================================
# 7. PREDICTION FUNCTION
# =============================================================================
def predict_crops_categorical(trained_models, best_model, X_encoded, df_soil, df_weather_processed,
                            nitrogen_status=None, phosphorus_status=None, potassium_status=None, 
                            ph_status=None, temperature_status=None, humidity_status=None, 
                            rainfall_status=None, state=None, district=None, block=None, top_n=5):
    """
    Predict crop recommendations using categorical inputs
    
    Two input methods:
    1. Direct categorical input: ALL NPK + pH parameters required (temperature, humidity, rainfall auto-filled from weather data)
    2. Location-based: state, district, block required (gets soil + weather data automatically)
    """
    
    # Method 1: Location-based prediction
    if state and district and block:
        # Find soil data for the specified location
        soil_data = df_soil[
            (df_soil['State'].str.upper() == state.upper()) & 
            (df_soil['District'].str.upper() == district.upper()) & 
            (df_soil['Block'].str.upper() == block.upper())
        ]
        
        if soil_data.empty:
            print(f"❌ No soil data found for {state}, {district}, {block}")
            print("Available locations (first 5):")
            print(df_soil[['State', 'District', 'Block']].head())
            return
        
        # Get weather data for the state
        weather_data = df_weather_processed[df_weather_processed['State'].str.upper() == state.upper()]
        if weather_data.empty:
            print(f"❌ No weather data found for state: {state}")
            print("Available states:")
            print(df_weather_processed['State'].unique())
            return
            
        soil_row = soil_data.iloc[0]
        weather_row = weather_data.iloc[0]
        
        input_data = {
            'Nitrogen_Status_crop': nitrogen_status or 'Medium',  # User can specify crop conditions
            'Phosphorous_Status_crop': phosphorus_status or 'Medium', 
            'Potassium_Status_crop': potassium_status or 'Medium',
            'pH_Status_crop': ph_status or 'Neutral',
            'Temperature_Status_crop': temperature_status or weather_row['Temperature_Status_weather'],  # Use weather data
            'Humidity_Status_crop': humidity_status or weather_row['Humidity_Status_weather'],  # Use weather data
            'Rainfall_Status_crop': rainfall_status or weather_row['Rainfall_Status_weather'],  # Use weather data
            'Temperature_Status_weather': weather_row['Temperature_Status_weather'],  # From weather data
            'Humidity_Status_weather': weather_row['Humidity_Status_weather'],  # From weather data
            'Rainfall_Status_weather': weather_row['Rainfall_Status_weather'],  # From weather data
            'Nitrogen_Status_soil': soil_row['Nitrogen_Status'],
            'Phosphorous_Status_soil': soil_row['Phosphorous_Status'],
            'Potassium_Status_soil': soil_row['Potassium_Status'],
            'pH_Status_soil': soil_row['pH_Status'],
            'OC_Status': soil_row['OC_Status'],
            'EC_Status': soil_row['EC_Status']
        }
        print(f"🌍 Location-based prediction for: {state}, {district}, {block}")
        print(f"🌱 Using soil data from location + weather data from state")
        
    # Method 2: Direct categorical input - Only NPK + pH required, weather auto-filled
    else:
        # Check if required NPK and pH parameters are provided
        required_params = [nitrogen_status, phosphorus_status, potassium_status, ph_status]
        param_names = ['nitrogen_status', 'phosphorus_status', 'potassium_status', 'ph_status']
        
        if any(param is None for param in required_params):
            print("❌ ERROR: For direct categorical prediction, NPK + pH parameters are required:")
            print("   nitrogen_status, phosphorus_status, potassium_status, ph_status")
            print("   Each should be 'Low', 'Medium', or 'High' (pH: 'Acidic', 'Neutral', 'Alkaline')")
            print("   Temperature, humidity, rainfall will be set to average values")
            return
        
        input_data = {
            'Nitrogen_Status_crop': nitrogen_status,
            'Phosphorous_Status_crop': phosphorus_status,
            'Potassium_Status_crop': potassium_status, 
            'pH_Status_crop': ph_status,
            'Temperature_Status_crop': temperature_status or 'Medium',  # Default if not provided
            'Humidity_Status_crop': humidity_status or 'Medium',  # Default if not provided
            'Rainfall_Status_crop': rainfall_status or 'Medium',  # Default if not provided
            'Temperature_Status_weather': 'Medium',  # Average weather conditions
            'Humidity_Status_weather': 'Medium',
            'Rainfall_Status_weather': 'Medium',
            # Use average soil conditions when no location provided
            'Nitrogen_Status_soil': 'Medium',  
            'Phosphorous_Status_soil': 'Medium',
            'Potassium_Status_soil': 'Medium',
            'pH_Status_soil': 'Neutral',
            'OC_Status': 'Medium',
            'EC_Status': 'Non Saline'
        }
        print(f"🧪 Direct categorical prediction with NPK + pH (weather conditions: Medium)")
    
    print(f"📊 Input features: {input_data}")
    
    # Prepare input for prediction
    input_df = pd.DataFrame([input_data])
    input_encoded = pd.get_dummies(input_df, drop_first=False)
    
    # Align input features with training features
    for col in X_encoded.columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[X_encoded.columns]
    
    # Make prediction using best model
    best_model_obj = trained_models[best_model[0]]
    probabilities = best_model_obj.predict_proba(input_encoded)[0]
    classes = best_model_obj.classes_
    
    # Get top N recommendations
    top_indices = probabilities.argsort()[::-1][:top_n]

    print(f"\n🌾 Top {top_n} Crop Recommendations:")
    print("-" * 50)
    for i, idx in enumerate(top_indices, 1):
        crop = classes[idx]
        print(f"{i}. {crop}")

    return [classes[idx] for idx in top_indices]

# =============================================================================
# 8. MAIN EXECUTION
# =============================================================================
def main():
    set_global_seed(42)
    """Main function to run the complete pipeline"""
    print("🌾 CATEGORICAL CROP RECOMMENDATION SYSTEM")
    print("=" * 60)
    
    # Step 1: Load datasets
    df_crop, df_soil, df_weather = load_datasets()
    
    # Step 2: Convert numerical to categorical
    df_crop_categorical = convert_to_categorical(df_crop)
    
    # Step 3: Process weather data
    df_weather_processed = prepare_weather_data(df_weather)
    
    # Step 4: Merge crop, soil, and weather data
    merged_data = merge_crop_soil_data(df_crop_categorical, df_soil, df_weather_processed)
    
    # Step 5: Prepare data for ML
    X_encoded, y, merged_clean = prepare_ml_data(merged_data)
    
    # Step 6: Train models
    trained_models, best_model, X_encoded = train_models(X_encoded, y)
    
    # Step 7: Example predictions
    print("\n" + "=" * 60)
    print("🎯 EXAMPLE PREDICTIONS")
    print("=" * 60)
    
    # Example 1: Direct categorical input - Only NPK + pH required
    print("\n📌 Example 1: NPK + pH input (weather auto-filled)")
    predict_crops_categorical(trained_models, best_model, X_encoded, df_soil, df_weather_processed,
                             nitrogen_status='High', phosphorus_status='Medium', potassium_status='Low',
                             ph_status='Neutral')
    
    # Example 2: Location-based prediction - gets soil + weather data automatically
    print("\n📌 Example 2: Location-based prediction (soil + weather auto-filled)")
    # CHANGE THESE VALUES TO TEST DIFFERENT LOCATIONS:
    state = "Tamil Nadu"           # Change this
    district = "Erode"          # Change this  
    block = "GOPICHETTIPALAIYAM"        # Change this
    
    predict_crops_categorical(trained_models, best_model, X_encoded, df_soil, df_weather_processed,
                             state=state, district=district, block=block)

    
    print(f"\n✅ SYSTEM READY!")
    print(f"✅ Best Model: {best_model[0]} with {best_model[1]:.4f} accuracy")
    print(f"✅ Weather data incorporated - temperature, humidity, rainfall auto-filled")
    print(f"✅ Change state, district, block variables in code to test different locations")
    
    return trained_models, best_model, X_encoded, df_soil, df_weather_processed

# =============================================================================
# RUN THE SYSTEM
# =============================================================================
if __name__ == "__main__":
    trained_models, best_model, X_encoded, df_soil, df_weather_processed = main()
    
    # You can now call predict_crops_categorical() with different parameters
    # Example:
    # predict_crops_categorical(trained_models, best_model, X_encoded, df_soil, df_weather_processed,
    #                          state="Karnataka", district="Bagalakote", block="JAMKHANDI")
