from rapidfuzz import process, fuzz
import re
import json
import os
# =====================
# Load location aliases from location_hierarchy.json
# =====================
LOCATION_JSON_PATH = os.path.join(os.path.dirname(__file__), '../AgriVision/src/data/location_hierarchy.json')
try:
    with open(LOCATION_JSON_PATH, encoding='utf-8') as f:
        location_json = json.load(f)
        location_aliases = location_json.get('aliases', {})
except Exception as e:
    print(f"[WARN] Could not load location aliases: {e}")
    location_aliases = {}
"""
FastAPI backend for Categorical Crop Recommendation System
Exposes endpoints for location-based and direct NPK+pH input crop prediction
"""
from fastapi import FastAPI, HTTPException

from reverse_geocode_proxy import router as reverse_geocode_router
from pydantic import BaseModel, Field
from typing import Optional, List
import uvicorn
import traceback
from solution5_simplified import (
    load_datasets, convert_to_categorical, prepare_weather_data,
    merge_crop_soil_data, prepare_ml_data, train_models, predict_crops_categorical
)
import os
import joblib
from fastapi.middleware.cors import CORSMiddleware

# =====================
# FastAPI Setup
# =====================
app = FastAPI(title="Crop Recommendation API", description="Location-based and NPK+pH input endpoints.")

# Register the reverse geocode router for /reverse-geocode/
app.include_router(reverse_geocode_router)

# Add CORS middleware before any routes
origins = [
    "http://localhost:8080",
    "http://localhost:5173",
    "https://agrivision-ai-based-crop-recommender.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================
# Model/Encoder Saving/Loading
# =====================
MODEL_PATH = "crop_model.joblib"
ENCODER_PATH = "crop_encoder.joblib"
SOIL_PATH = "soil_df.joblib"
WEATHER_PATH = "weather_df.joblib"

def train_and_save():
    print("\n[INFO] Training model and saving artifacts...")
    df_crop, df_soil, df_weather = load_datasets()
    df_crop_categorical = convert_to_categorical(df_crop)
    df_weather_processed = prepare_weather_data(df_weather)
    merged_data = merge_crop_soil_data(df_crop_categorical, df_soil, df_weather_processed)
    X_encoded, y, merged_clean = prepare_ml_data(merged_data)
    trained_models, best_model, X_encoded = train_models(X_encoded, y)
    # Save best model, encoder (X_encoded columns), and dataframes
    joblib.dump((trained_models, best_model), MODEL_PATH)
    joblib.dump(list(X_encoded.columns), ENCODER_PATH)
    joblib.dump(df_soil, SOIL_PATH)
    joblib.dump(df_weather_processed, WEATHER_PATH)
    print("[INFO] Model and encoder saved!")
    return trained_models, best_model, X_encoded, df_soil, df_weather_processed

def load_all():
    print("[INFO] Loading model and encoder from disk...")
    trained_models, best_model = joblib.load(MODEL_PATH)
    encoder_columns = joblib.load(ENCODER_PATH)
    df_soil = joblib.load(SOIL_PATH)
    df_weather_processed = joblib.load(WEATHER_PATH)
    # Create empty DataFrame with encoder columns for alignment
    import pandas as pd
    X_encoded = pd.DataFrame(columns=encoder_columns)
    print("[INFO] Model and encoder loaded!")
    return trained_models, best_model, X_encoded, df_soil, df_weather_processed

# =====================
# Model Loading at Startup
# =====================
if not (os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH) and os.path.exists(SOIL_PATH) and os.path.exists(WEATHER_PATH)):
    trained_models, best_model, X_encoded, df_soil, df_weather_processed = train_and_save()
else:
    trained_models, best_model, X_encoded, df_soil, df_weather_processed = load_all()

# =====================
# Request Models
# =====================
class LocationRequest(BaseModel):
    state: str
    district: str
    block: str
    nitrogen_status: Optional[str] = Field(None, description="Low/Medium/High")
    phosphorus_status: Optional[str] = Field(None, description="Low/Medium/High")
    potassium_status: Optional[str] = Field(None, description="Low/Medium/High")
    ph_status: Optional[str] = Field(None, description="Acidic/Neutral/Alkaline")
    temperature_status: Optional[str] = None
    humidity_status: Optional[str] = None
    rainfall_status: Optional[str] = None
    top_n: int = 5

class NPKRequest(BaseModel):
    nitrogen_status: str
    phosphorus_status: str
    potassium_status: str
    ph_status: str
    temperature_status: Optional[str] = None
    humidity_status: Optional[str] = None
    rainfall_status: Optional[str] = None
    top_n: int = 5



class LocationPredictionResponse(BaseModel):
    nearest_state: str
    nearest_district: str
    nearest_block: str
    predictions: List[str]
    soil_data: dict = {}
    weather_data: Optional[dict] = None

# =====================
# Endpoints
# =====================
@app.post("/predict/location", response_model=LocationPredictionResponse)
def predict_location(req: LocationRequest):
    try:
        def apply_alias(name):
            if not name:
                return name
            # Try direct match (case-insensitive)
            for k, v in location_aliases.items():
                if name.strip().lower() == k.strip().lower():
                    return v
            return name

        def normalize_name(name):
            if not name:
                return ""
            name = name.lower()
            name = re.sub(r"[^a-z0-9 ]", "", name)
            name = re.sub(r"\b(district|urban|rural|block|taluk|taluka|mandal|city|subdistrict|sub-district|zone)\b", "", name)
            name = re.sub(r"\s+", " ", name).strip()
            return name

        # Normalize all names in the database
        df_soil_norm = df_soil.copy()
        df_soil_norm['State_norm'] = df_soil_norm['State'].apply(normalize_name)
        df_soil_norm['District_norm'] = df_soil_norm['District'].apply(normalize_name)
        df_soil_norm['Block_norm'] = df_soil_norm['Block'].apply(normalize_name)

        # Apply alias mapping before normalization
        state_aliased = apply_alias(req.state)
        district_aliased = apply_alias(req.district)
        block_aliased = apply_alias(req.block)

        # Normalize input
        state_in = normalize_name(state_aliased)
        district_in = normalize_name(district_aliased)
        block_in = normalize_name(block_aliased)

        # Fuzzy match state
        state_choices = df_soil_norm['State_norm'].unique().tolist()
        state_match, state_score, _ = process.extractOne(state_in, state_choices, scorer=fuzz.ratio)
        if state_score < 80:
            raise HTTPException(status_code=404, detail=f"State '{req.state}' not found (closest: '{state_match}', score: {state_score})")
        df_state = df_soil_norm[df_soil_norm['State_norm'] == state_match]
        matched_state = df_state.iloc[0]['State'] if not df_state.empty else req.state

        # Fetch weather data for matched state
        weather_data = None
        try:
            state_weather = df_weather_processed[df_weather_processed['State'].str.lower() == matched_state.lower()]
            if state_weather.empty:
                state_weather = df_weather_processed[df_weather_processed['State'] == matched_state]
            if not state_weather.empty:
                latest_year = state_weather['year'].max()
                weather_row = state_weather[state_weather['year'] == latest_year].iloc[0]
                weather_data = {
                    "temperature": float(weather_row.get('Avg_Temp_C', 0)),
                    "rainfall": float(weather_row.get('Total_Rainfall_mm', 0)),
                    "year": int(latest_year)
                }
        except Exception as e:
            print(f"[WARN] Could not extract weather_data: {e}")

        # Fuzzy match district within state
        district_choices = df_state['District_norm'].unique().tolist()
        district_match, district_score, _ = process.extractOne(district_in, district_choices, scorer=fuzz.ratio)
        if district_score < 40:
            # Try partial/substring match as fallback
            partial_matches = [d for d in district_choices if district_in in d or d in district_in]
            if partial_matches:
                district_match = partial_matches[0]
            else:
                # Try fuzzy match for any district containing the input as a substring
                substring_matches = [d for d in district_choices if district_in and district_in in d]
                if substring_matches:
                    district_match = substring_matches[0]
                else:
                    raise HTTPException(status_code=404, detail=f"District '{req.district}' not found in state '{req.state}' (closest: '{district_match}', score: {district_score})")
        # Always use the matched district for downstream
        df_district = df_state[df_state['District_norm'] == district_match]
        matched_district = df_district.iloc[0]['District'] if not df_district.empty else req.district

        # Fuzzy match block within district
        block_choices = df_district['Block_norm'].unique().tolist()
        if not block_in:
            # If no block provided, pick the first block in the district
            chosen_row = df_district.iloc[[0]]
            chosen_block = chosen_row.iloc[0]['Block']
            matched_block = chosen_block
        else:
            block_match, block_score, _ = process.extractOne(block_in, block_choices, scorer=fuzz.ratio)
            if block_score < 40:
                # Try partial/substring match as fallback
                partial_matches = [b for b in block_choices if block_in and (block_in in b or b in block_in)]
                if partial_matches:
                    block_match = partial_matches[0]
                    chosen_row = df_district[df_district['Block_norm'] == block_match]
                    chosen_block = chosen_row.iloc[0]['Block']
                    matched_block = chosen_block
                else:
                    import random
                    chosen_row = df_district.sample(n=1, random_state=None)
                    chosen_block = chosen_row.iloc[0]['Block']
                    matched_block = chosen_block
            else:
                chosen_row = df_district[df_district['Block_norm'] == block_match]
                chosen_block = chosen_row.iloc[0]['Block']
                matched_block = chosen_block

        soil_row = chosen_row.iloc[0] if not chosen_row.empty else None
        # Extract categorical NPK + pH status values
        soil_data = {}
        if soil_row is not None:
            try:
                soil_data = {
                    "nitrogen": str(soil_row.get('Nitrogen_Status', 'Unknown')),
                    "phosphorous": str(soil_row.get('Phosphorous_Status', 'Unknown')),
                    "potassium": str(soil_row.get('Potassium_Status', 'Unknown')),
                    "ph": str(soil_row.get('pH_Status', 'Unknown')),
                }
            except Exception as e:
                print(f"[WARN] Could not extract categorical soil_data: {e}")

        # Fetch weather data for matched state
        weather_data = None
        try:
            state_weather = df_weather_processed[df_weather_processed['State'].str.lower() == matched_state.lower()]
            if state_weather.empty:
                state_weather = df_weather_processed[df_weather_processed['State'] == matched_state]
            if not state_weather.empty:
                latest_year = state_weather['year'].max()
                weather_row = state_weather[state_weather['year'] == latest_year].iloc[0]
                weather_data = {
                    "temperature": float(weather_row.get('Avg_Temp_C', 0)),
                    "rainfall": float(weather_row.get('Total_Rainfall_mm', 0)),
                    "year": int(latest_year)
                }
        except Exception as e:
            print(f"[WARN] Could not extract weather_data: {e}")

        # Use the matched block for prediction
        result = predict_crops_categorical(
            trained_models, best_model, X_encoded, df_soil, df_weather_processed,
            nitrogen_status=req.nitrogen_status,
            phosphorus_status=req.phosphorus_status,
            potassium_status=req.potassium_status,
            ph_status=req.ph_status,
            temperature_status=req.temperature_status,
            humidity_status=req.humidity_status,
            rainfall_status=req.rainfall_status,
            state=matched_state, district=matched_district, block=matched_block,
            top_n=req.top_n
        )
        if not result:
            raise HTTPException(status_code=404, detail="No prediction could be made for the given location.")
        return LocationPredictionResponse(
            nearest_state=matched_state,
            nearest_district=matched_district,
            nearest_block=matched_block,
            predictions=result,
            soil_data=soil_data,
            weather_data=weather_data
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/npk", response_model=List[str])
def predict_npk(req: NPKRequest):
    try:
        result = predict_crops_categorical(
            trained_models, best_model, X_encoded, df_soil, df_weather_processed,
            nitrogen_status=req.nitrogen_status,
            phosphorus_status=req.phosphorus_status,
            potassium_status=req.potassium_status,
            ph_status=req.ph_status,
            temperature_status=req.temperature_status,
            humidity_status=req.humidity_status,
            rainfall_status=req.rainfall_status,
            top_n=req.top_n
        )
        if not result:
            raise HTTPException(status_code=400, detail="NPK + pH input insufficient for prediction.")
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# =====================
# Run with: uvicorn main_api:app --reload
# =====================
if __name__ == "__main__":
    uvicorn.run("main_api:app", host="0.0.0.0", port=8000, reload=True)
