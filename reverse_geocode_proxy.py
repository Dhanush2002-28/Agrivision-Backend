import requests
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter()

@router.get("/reverse-geocode/")
def reverse_geocode(lat: float, lon: float):
    url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&zoom=10&addressdetails=1"
    headers = {"User-Agent": "AgriVision/1.0 (contact@example.com)"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reverse geocoding failed: {str(e)}")
