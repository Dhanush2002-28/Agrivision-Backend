from fastapi import FastAPI, HTTPException, Request, Query
import pandas as pd
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import datetime

app = FastAPI()

# Load soil lookup and model at startup
soil_df = pd.read_csv("data/processed/soil_lookup.csv")
weather_df = pd.read_csv("data/processed/weather_lookup.csv")
model = joblib.load("data/processed/crop_recommendation_lgbm_augmented.pkl")

crop_label_df = pd.read_csv("data/processed/crop_label_map.csv", index_col=0)
CROP_LABEL_MAPPING = {v: k for k, v in crop_label_df.iloc[:, 0].to_dict().items()}

# Add reverse geocode endpoint
@app.get("/reverse-geocode")
def reverse_geocode(lat: float, lon: float):
    """Simple reverse geocoding using fallback data"""
    try:
        return {
            "address": {
                "postcode": "560095",     
                "state": "Karnataka",     
                "county": "Bengaluru Urban",
                "city": "Bengaluru", 
                "suburb": "Central Bengaluru"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reverse geocoding failed: {str(e)}")

@app.post("/recommend-crop")  
def recommend_crop(request_data: dict):
    """Endpoint that matches frontend expectations"""
    pincode = request_data.get("pincode")
    if not pincode:
        raise HTTPException(status_code=400, detail="Pincode is required")
    
    result = recommend_crops(pincode)
    
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    
    return {
        "pincode": result["pincode"],
        "location": {
            "state": result["state"],
            "district": result["district"],
            "block": result["block"]
        },
        "soil_data": result.get("soil_data", {}),  
        "weather_data": result.get("weather_data", {}),  
        "recommended_crops": [
            {
                "crop": crop["crop"],
                "probability": crop["confidence"] / 100
            }
            for crop in result["recommended_crops"]
        ]
    }

REGIONAL_CROPS = {
    "karnataka": {
        "suitable": [
            "Rice", "Maize", "Cotton", "Sugarcane", "Coffee", "Coconut", 
            "Banana", "Mango", "Papaya", "Grapes", "Pomegranate", "Orange",
            "Jowar", "Ragi", "Tur", "MungBean", "Sesame", "Tapioca"
        ],
        "unsuitable": ["Garlic", "Cumin", "ChickPea", "Lentil"]  
    },
    "goa": {
        "suitable": [
            "Rice", "Coconut", "Cashew", "Banana", "Mango", "Papaya",
            "Pineapple", "Orange", "Brinjal", "Chilli", "Okra"
        ],
        "unsuitable": ["Cotton", "Sugarcane", "Wheat", "Garlic", "Cumin"]
    },
    "gujarat": {
        "suitable": [
            "Cotton", "Sugarcane", "Garlic", "Cumin", "ChickPea", "Lentil",
            "Maize", "Jowar", "Sesame", "Banana", "Mango", "Pomegranate"
        ],
        "unsuitable": ["Coffee", "Cashew", "Coconut", "Pineapple"]
    },
    "tamil nadu": {
        "suitable": [
            "Rice", "Sugarcane", "Cotton", "Maize", "Banana", "Mango",
            "Coconut", "Tapioca", "Turmeric", "Chilli", "Onion", "Tomato"
        ],
        "unsuitable": ["Coffee", "Garlic", "Cumin"]
    }
}

SEASONAL_CROPS = {
    "kharif": ["Rice", "Cotton", "Sugarcane", "Maize", "Jowar", "Banana"],
    "rabi": ["Wheat", "ChickPea", "Lentil", "Mustard", "Garlic", "Onion"],
    "zaid": ["Watermelon", "Muskmelon", "Cucumber", "Fodder crops"],
    "perennial": ["Coconut", "Mango", "Coffee", "Cashew", "Grapes", "Pomegranate"]
}

def get_current_season():
    """Determine current agricultural season"""
    month = datetime.datetime.now().month
    if month in [6, 7, 8, 9, 10]:  # June to October
        return "kharif"
    elif month in [11, 12, 1, 2, 3, 4]:  # November to April
        return "rabi"
    else:  # May
        return "zaid"

@app.get("/recommend_crops")
def recommend_crops(pincode: str):
    # Function to check if pincode exists in comma-separated string
    def pincode_match(pincode_str, target_pincode):
        if pd.isna(pincode_str):
            return False
        # Split by comma and strip whitespace, then check if target exists
        pincodes = [p.strip() for p in str(pincode_str).split(',')]
        return target_pincode in pincodes
    
    # Lookup soil data for pincode using string matching
    soil_row = soil_df[soil_df['Pincode'].apply(lambda x: pincode_match(x, pincode))]
    
    if soil_row.empty:
        # Show sample pincodes for debugging
        sample_pincodes = []
        for pc_str in soil_df['Pincode'].head(10):  # Check more rows
            if pd.notna(pc_str):
                sample_pincodes.extend([p.strip() for p in str(pc_str).split(',')])
        return {"error": f"Pincode {pincode} not found. Sample available pincodes: {sample_pincodes[:20]}"}
    
    # Get soil data
    soil_data = soil_row.iloc[0]
    state = soil_data['State']
    
    # Fix case sensitivity for state matching
    state_weather = weather_df[weather_df['State'].str.lower() == state.lower()]
    if state_weather.empty:
        # Try exact match if lowercase doesn't work
        state_weather = weather_df[weather_df['State'] == state]
        if state_weather.empty:
            return {"error": f"No weather data found for state: {state}. Available states: {weather_df['State'].unique().tolist()}"}
    
    # Get the latest year data for this state
    latest_year = state_weather['year'].max()
    weather_data = state_weather[state_weather['year'] == latest_year].iloc[0]
    
    # Prepare features for model prediction with proper feature names
    feature_names = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall']
    features = [
        float(soil_data['Total Nitrogen_x']),      # Nitrogen
        float(soil_data['Total Phosphorous_x']),   # Phosphorus  
        float(soil_data['Total Potassium_x']),     # Potassium
        float(weather_data['Avg_Temp_C']),         # Temperature
        75.0,                                      # Humidity (default value)
        float(soil_data['Total pH_x']),            # pH_Value
        float(weather_data['Total_Rainfall_mm'])   # Rainfall
    ]
    
    # Create DataFrame with proper feature names to avoid warning
    feature_df = pd.DataFrame([features], columns=feature_names)
    
    # Predict crops
    try:
        # Get prediction probabilities for all crops
        probabilities = model.predict_proba(feature_df)[0]
        crop_codes = model.classes_
        
        # Get state for regional filtering
        state_name = str(state).lower()
        
        # Apply regional filtering
        filtered_recommendations = []
        regional_data = REGIONAL_CROPS.get(state_name, {})
        suitable_crops = regional_data.get("suitable", [])
        unsuitable_crops = regional_data.get("unsuitable", [])
        
        # Get current season and seasonal crops
        current_season = get_current_season()
        seasonal_crops = SEASONAL_CROPS.get(current_season, []) + SEASONAL_CROPS.get("perennial", [])
        
        # Sort all crops by probability
        sorted_indices = probabilities.argsort()[::-1]
        
        for i in sorted_indices:
            crop_code = crop_codes[i]
            if isinstance(crop_code, (int, np.integer)):
                crop_name = CROP_LABEL_MAPPING.get(int(crop_code), f"Unknown Crop {crop_code}")
            else:
                crop_name = str(crop_code)
            
            # Skip if crop is explicitly unsuitable for this region
            if unsuitable_crops and crop_name in unsuitable_crops:
                continue
            
            # Boost probability if crop is regionally suitable
            original_prob = float(probabilities[i])
            if suitable_crops and crop_name in suitable_crops:
                # Boost suitable crops by 20%
                boosted_prob = min(original_prob * 1.2, 1.0)
            else:
                boosted_prob = original_prob
            
            # Boost seasonal crops
            if crop_name in seasonal_crops:
                boosted_prob = min(boosted_prob * 1.3, 1.0)  # 30% boost for seasonal crops
            else:
                boosted_prob *= 0.8  # Reduce off-season crops
            
            filtered_recommendations.append({
                "crop": crop_name,
                "confidence": round(boosted_prob * 100, 2),
                "original_confidence": round(original_prob * 100, 2),
                "regionally_suitable": crop_name in suitable_crops if suitable_crops else True
            })
            
            # Stop when we have 4 recommendations
            if len(filtered_recommendations) >= 4:
                break
        
        return {
            "pincode": pincode,
            "state": str(state),
            "district": str(soil_data['District']), 
            "block": str(soil_data['Block']),
            "soil_data": {
                "nitrogen": float(soil_data['Total Nitrogen_x']),
                "phosphorous": float(soil_data['Total Phosphorous_x']),
                "potassium": float(soil_data['Total Potassium_x']),
                "ph": float(soil_data['Total pH_x'])
            },
            "weather_data": {
                "temperature": float(weather_data['Avg_Temp_C']),
                "rainfall": float(weather_data['Total_Rainfall_mm']),
                "year": int(latest_year)
            },
            "recommended_crops": filtered_recommendations[:4]  # Top 4 after filtering
        }
        
    except Exception as e:
        return {"error": f"Model prediction failed: {str(e)}"}

# Debug endpoint to verify crop mapping
@app.get("/debug/crop-mapping")
def debug_crop_mapping():
    """Debug endpoint to see the loaded crop mapping"""
    return {
        "total_crops": len(CROP_LABEL_MAPPING),
        "sample_mapping": dict(list(CROP_LABEL_MAPPING.items())[:10]),
        "all_crops": list(CROP_LABEL_MAPPING.values())
    }

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)