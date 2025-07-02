import pandas as pd

# Load existing data
X = pd.read_csv('processed/X_train.csv')
y = pd.read_csv('processed/y_train.csv')

# Add geographic features based on known cropping patterns
def add_geographic_features(X):
    X_geo = X.copy()
    
    # Add region-based features
    # These could be based on latitude bands, agro-climatic zones, etc.
    
    # Example: Temperature-based regions
    X_geo['Climate_Zone'] = 0  # Default
    X_geo.loc[X_geo['Temperature'] < 20, 'Climate_Zone'] = 1  # Cool climate
    X_geo.loc[(X_geo['Temperature'] >= 20) & (X_geo['Temperature'] < 30), 'Climate_Zone'] = 2  # Moderate
    X_geo.loc[X_geo['Temperature'] >= 30, 'Climate_Zone'] = 3  # Hot climate
    
    # Rainfall-based zones
    X_geo['Rainfall_Zone'] = 0
    X_geo.loc[X_geo['Rainfall'] < 100, 'Rainfall_Zone'] = 1  # Arid
    X_geo.loc[(X_geo['Rainfall'] >= 100) & (X_geo['Rainfall'] < 200), 'Rainfall_Zone'] = 2  # Semi-arid
    X_geo.loc[X_geo['Rainfall'] >= 200, 'Rainfall_Zone'] = 3  # Humid
    
    return X_geo

X_enhanced = add_geographic_features(X)
X_enhanced.to_csv('processed/X_train_geographic.csv', index=False)
print("Enhanced training data with geographic features saved!")