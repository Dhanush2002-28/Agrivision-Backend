"""
LOCATION TO BLOCK RESOLVER: Practical solution for getting block data from coordinates
Implements multiple strategies to resolve coordinates to block-level information
"""
import pandas as pd
import math
import requests
import json
from typing import Dict, Optional, List, Tuple

class LocationToBlockResolver:
    """
    Resolve coordinates to block-level information using multiple strategies
    """
    
    def __init__(self, soil_data_path: str):
        """Initialize with soil dataset"""
        self.df_soil = pd.read_csv(soil_data_path)
        self.block_lookup = self._create_block_lookup()
        
    def _create_block_lookup(self) -> Dict:
        """Create lookup tables for efficient block resolution"""
        lookup = {
            'by_district': {},
            'block_coordinates': {}  # You'd populate this with real coordinates
        }
        
        # Create district -> blocks mapping
        for _, row in self.df_soil.iterrows():
            state = row['State']
            district = row['District'] 
            block = row['Block']
            
            key = f"{state}|{district}"
            if key not in lookup['by_district']:
                lookup['by_district'][key] = []
            
            if block not in lookup['by_district'][key]:
                lookup['by_district'][key].append(block)
        
        # Sample coordinates for major blocks (you'd get these from geocoding)
        # This is just example data - you'd need to populate with real coordinates
        lookup['block_coordinates'] = {
            ('Karnataka', 'Bagalakote', 'JAMKHANDI'): (16.1848, 75.6791),
            ('Karnataka', 'Bagalakote', 'BADAMI'): (15.9149, 75.6767),
            ('Gujarat', 'Ahmedabad', 'DASKROI'): (23.0225, 72.5714),
            ('Tamil Nadu', 'Chennai', 'THIRUVALLUR'): (13.1068, 80.0982),
            # Add more coordinates as you collect them
        }
        
        return lookup
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points on Earth in kilometers"""
        R = 6371  # Earth's radius in kilometers
        
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    def get_location_from_coordinates(self, lat: float, lon: float) -> Optional[Dict]:
        """Get basic location info from coordinates using free APIs"""
        try:
            # Try alternative geocoding services
            apis_to_try = [
                {
                    'name': 'LocationIQ',
                    'url': f'https://us1.locationiq.com/v1/reverse.php?key=YOUR_API_KEY&lat={lat}&lon={lon}&format=json',
                    'needs_key': True
                },
                {
                    'name': 'MapBox',
                    'url': f'https://api.mapbox.com/geocoding/v5/mapbox.places/{lon},{lat}.json?access_token=YOUR_TOKEN',
                    'needs_key': True
                },
                {
                    'name': 'BigDataCloud (Free)',
                    'url': f'https://api.bigdatacloud.net/data/reverse-geocode-client?latitude={lat}&longitude={lon}&localityLanguage=en',
                    'needs_key': False
                }
            ]
            
            # Try BigDataCloud first (free, no key needed)
            response = requests.get(apis_to_try[2]['url'], timeout=5)
            if response.status_code == 200:
                data = response.json()
                return {
                    'state': data.get('principalSubdivision', ''),
                    'district': data.get('locality', ''),
                    'city': data.get('city', ''),
                    'country': data.get('countryName', ''),
                    'source': 'BigDataCloud'
                }
        except Exception as e:
            print(f"API call failed: {e}")
        
        return None
    
    def find_nearest_block_by_proximity(self, lat: float, lon: float, max_distance_km: float = 50) -> Optional[Dict]:
        """Find nearest block using proximity matching"""
        nearest_block = None
        min_distance = float('inf')
        
        for (state, district, block), (block_lat, block_lon) in self.block_lookup['block_coordinates'].items():
            distance = self._haversine_distance(lat, lon, block_lat, block_lon)
            
            if distance < min_distance and distance <= max_distance_km:
                min_distance = distance
                nearest_block = {
                    'state': state,
                    'district': district,
                    'block': block,
                    'distance_km': round(distance, 2),
                    'method': 'proximity'
                }
        
        return nearest_block
    
    def get_most_common_block_in_district(self, state: str, district: str) -> Optional[Dict]:
        """Get the most common block in a district (fallback strategy)"""
        key = f"{state}|{district}"
        
        if key in self.block_lookup['by_district']:
            # Filter soil data for this district to get block with most data points
            district_data = self.df_soil[
                (self.df_soil['State'].str.upper() == state.upper()) & 
                (self.df_soil['District'].str.upper() == district.upper())
            ]
            
            if len(district_data) > 0:
                # Get most common block (block with most soil data points)
                most_common_block = district_data['Block'].value_counts().index[0]
                
                return {
                    'state': state,
                    'district': district,
                    'block': most_common_block,
                    'method': 'most_common_in_district',
                    'data_points': district_data[district_data['Block'] == most_common_block].shape[0]
                }
        
        return None
    
    def get_available_blocks_in_district(self, state: str, district: str) -> List[str]:
        """Get all available blocks in a district for manual selection"""
        key = f"{state}|{district}"
        return self.block_lookup['by_district'].get(key, [])
    
    def resolve_coordinates_to_block(self, lat: float, lon: float) -> Dict:
        """
        Main function to resolve coordinates to block using multiple strategies
        Returns comprehensive result with all attempted methods
        """
        result = {
            'coordinates': {'lat': lat, 'lon': lon},
            'location_info': None,
            'block_result': None,
            'available_methods': [],
            'recommendations': []
        }
        
        # Step 1: Get basic location info
        print(f"🌍 Resolving coordinates: {lat}, {lon}")
        location_info = self.get_location_from_coordinates(lat, lon)
        result['location_info'] = location_info
        
        if location_info:
            print(f"✅ Location: {location_info.get('state')}, {location_info.get('district')}")
        
        # Step 2: Try proximity matching
        print("🔍 Trying proximity matching...")
        proximity_result = self.find_nearest_block_by_proximity(lat, lon)
        if proximity_result:
            result['block_result'] = proximity_result
            result['available_methods'].append('proximity')
            print(f"✅ Found nearby block: {proximity_result['block']} ({proximity_result['distance_km']} km away)")
            return result
        
        # Step 3: Try district-based fallback
        if location_info and location_info.get('state') and location_info.get('district'):
            print("🔍 Trying district-based fallback...")
            district_result = self.get_most_common_block_in_district(
                location_info['state'], 
                location_info['district']
            )
            if district_result:
                result['block_result'] = district_result
                result['available_methods'].append('district_fallback')
                print(f"✅ Using most common block in district: {district_result['block']}")
                return result
        
        # Step 4: Manual selection option
        if location_info and location_info.get('state') and location_info.get('district'):
            available_blocks = self.get_available_blocks_in_district(
                location_info['state'], 
                location_info['district']
            )
            if available_blocks:
                result['available_methods'].append('manual_selection')
                result['recommendations'] = available_blocks[:10]  # Top 10 options
                print(f"📋 Manual selection available: {len(available_blocks)} blocks in district")
        
        # No automatic resolution possible
        if not result['block_result']:
            print("❌ Could not automatically resolve block - manual selection required")
        
        return result

def create_location_api_integration():
    """Create a practical API integration example"""
    print("\n🔗 LOCATION API INTEGRATION EXAMPLE")
    print("=" * 60)
    
    api_code = '''
# Flask API endpoint example
from flask import Flask, request, jsonify

app = Flask(__name__)
resolver = LocationToBlockResolver('path/to/soil_data.csv')

@app.route('/api/resolve-location', methods=['POST'])
def resolve_location():
    data = request.get_json()
    lat = data.get('latitude')
    lon = data.get('longitude')
    
    if not lat or not lon:
        return jsonify({'error': 'Latitude and longitude required'}), 400
    
    result = resolver.resolve_coordinates_to_block(lat, lon)
    
    if result['block_result']:
        # Success - got block data
        return jsonify({
            'success': True,
            'location': result['location_info'],
            'block': result['block_result'],
            'method': result['block_result']['method']
        })
    elif result['recommendations']:
        # Need manual selection
        return jsonify({
            'success': False,
            'needs_manual_selection': True,
            'location': result['location_info'],
            'available_blocks': result['recommendations']
        })
    else:
        # Complete failure
        return jsonify({
            'success': False,
            'error': 'Location not found in coverage area'
        })

@app.route('/api/get-blocks-in-district', methods=['GET'])
def get_blocks_in_district():
    state = request.args.get('state')
    district = request.args.get('district')
    
    blocks = resolver.get_available_blocks_in_district(state, district)
    return jsonify({'blocks': blocks})
'''
    
    print("Created Flask API integration example!")
    print("Key endpoints:")
    print("✅ POST /api/resolve-location - Main resolution endpoint")
    print("✅ GET /api/get-blocks-in-district - Manual selection helper")

def main():
    """Demonstrate the location to block resolver"""
    print("🎯 LOCATION TO BLOCK RESOLVER DEMO")
    print("=" * 60)
    
    # Initialize resolver
    soil_data_path = r'C:\Users\DHANUSH C\OneDrive\Desktop\AgriVision_Project\Project\Datasets\SoilData_DominantCategories.csv'
    resolver = LocationToBlockResolver(soil_data_path)
    
    # Test cases
    test_coordinates = [
        (16.1848, 75.6791),  # Bagalakote, Karnataka
        (23.0225, 72.5714),  # Ahmedabad, Gujarat  
        (13.0827, 80.2707),  # Chennai, Tamil Nadu
    ]
    
    for lat, lon in test_coordinates:
        print(f"\n{'='*50}")
        result = resolver.resolve_coordinates_to_block(lat, lon)
        
        if result['block_result']:
            block = result['block_result']
            print(f"🎉 SUCCESS: {block['state']}, {block['district']}, {block['block']}")
            print(f"   Method: {block['method']}")
        else:
            print("❌ No automatic resolution - manual selection needed")
            if result['recommendations']:
                print(f"   Available blocks: {', '.join(result['recommendations'][:3])}...")
    
    # Create API integration example
    create_location_api_integration()
    
    print(f"\n✅ RESOLVER DEMO COMPLETE!")
    print("🎯 NEXT STEPS:")
    print("1. Collect real coordinates for all blocks")
    print("2. Integrate with your preferred geocoding API")
    print("3. Implement in your Flask backend")
    print("4. Add caching for performance")

if __name__ == "__main__":
    main()
