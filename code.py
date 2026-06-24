import streamlit as st
import folium
from streamlit_folium import st_folium
import requests
from geopy.geocoders import Nominatim
import statistics
import pandas as pd

# Set page layout to wide
st.set_page_config(page_title="Multi-Origin Routing App", layout="wide")

st.title("🗺️ Multi-Origin Routing & Fair Meetup App")
st.markdown("Find the most fair and optimal meeting spots for multiple starting locations.")

# --- Initialization ---
if 'origins' not in st.session_state:
    st.session_state.origins = {} # name -> coords
if 'search_active' not in st.session_state:
    st.session_state.search_active = False
if 'results' not in st.session_state:
    st.session_state.results = []

geolocator = Nominatim(user_agent="multi_origin_streamlit_app")
colors = ["blue", "green", "purple", "orange", "darkred", "cadetblue"]

# --- Helper Functions ---
@st.cache_data(show_spinner=False)
def geocode_address(address):
    try:
        location = geolocator.geocode(address)
        if location:
            return location.latitude, location.longitude
    except Exception as e:
        import traceback; traceback.print_exc()
        st.error(f"Geocoding error for {address}: {e}")
    return None

@st.cache_data(show_spinner=False)
def get_route_info(origin_coords, dest_coords):
    url = f"http://router.project-osrm.org/route/v1/driving/{origin_coords[1]},{origin_coords[0]};{dest_coords[1]},{dest_coords[0]}?geometries=geojson"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data.get("code") == "Ok":
                route = data["routes"][0]
                coordinates = route["geometry"]["coordinates"]
                path = [(lat, lon) for lon, lat in coordinates]
                return path, route["distance"]
    except Exception as e:
        import traceback; traceback.print_exc()
    return None, None

def search_destinations(keyword, origin_data, similarity_threshold=0.15, kakao_api_key=None):
    dest_data = []
    if not origin_data:
        return dest_data
        
    avg_lat = sum(c[0] for _, c in origin_data) / len(origin_data)
    avg_lon = sum(c[1] for _, c in origin_data) / len(origin_data)
    radius = 15000 
    
    # Initialize search status
    st.session_state.search_status = "Unknown"
    
    # 1. Try our local SpotSync AI PostGIS Semantic Search Server!
    try:
        api_url = "http://127.0.0.1:8000/search"
        payload = {
            "query": keyword,
            "user_latitude": avg_lat,
            "user_longitude": avg_lon,
            "radius_meters": float(radius),
            "similarity_threshold": similarity_threshold,
            "top_k": 15
        }
        resp = requests.post(api_url, json=payload, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results", [])
            for r in results:
                dest_data.append({
                    "name": r.get("name", "Unknown Place"),
                    "address": r.get("address") or r.get("description") or "Unknown Address",
                    "coords": (r.get("latitude", 0.0), r.get("longitude", 0.0))
                })
            if dest_data:
                st.session_state.search_status = f"🤖 SpotSync AI Local Semantic Search Server (Found {len(dest_data)} matches)"
                return dest_data
            else:
                st.session_state.search_status = "🤖 SpotSync AI Server is ONLINE, but returned 0 results within 15km."
    except Exception as e:
        import traceback; traceback.print_exc()
        
    # 2. Try Kakao Local API for semantic search (if API key is provided or found in .env)
    if not kakao_api_key or kakao_api_key.strip() == "":
        import os
        if os.path.exists(".env"):
            with open(".env", "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip().startswith("KAKAO_API_KEY="):
                        kakao_api_key = line.split("=", 1)[1].strip()
                        break
                        
    if kakao_api_key and kakao_api_key.strip() != "":
        url = "https://dapi.kakao.com/v2/local/search/keyword.json"
        headers = {"Authorization": f"KakaoAK {kakao_api_key}"}
        
        # Map standard generic terms to Kakao's optimal Korean keywords
        kakao_keyword_map = {
            "restaurant": "음식점", "식당": "음식점", "맛집": "맛집",
            "cafe": "카페", "coffee": "카페",
            "pub": "술집", "bar": "술집",
            "hospital": "병원", "pharmacy": "약국",
            "convenience store": "편의점", "mart": "마트"
        }
        
        # Map categories to Kakao Category Group Codes for better semantic filtering
        kakao_category_map = {
            "restaurant": "FD6", "식당": "FD6", "음식점": "FD6", "맛집": "FD6",
            "cafe": "CE7", "카페": "CE7", "커피": "CE7",
            "convenience store": "CS2", "편의점": "CS2",
            "mart": "MT1", "마트": "MT1",
            "hospital": "HP8", "병원": "HP8",
            "pharmacy": "PM9", "약국": "PM9"
        }
        
        search_query = kakao_keyword_map.get(keyword.lower(), keyword)
        category_code = kakao_category_map.get(keyword.lower(), "")
        
        params = {
            "query": search_query,
            "y": avg_lat,
            "x": avg_lon,
            "radius": radius,
            "size": 15
        }
        
        if category_code:
            params["category_group_code"] = category_code
            
        try:
            # Kakao returns max 15 results per page. Paginate to get 20.
            for page in [1, 2]:
                params["page"] = page
                resp = requests.get(url, headers=headers, params=params, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    for place in data.get("documents", []):
                        dest_data.append({
                            "name": place.get("place_name"),
                            "address": place.get("road_address_name") or place.get("address_name"),
                            "coords": (float(place["y"]), float(place["x"]))
                        })
                        if len(dest_data) >= 20:
                            break
                    if len(dest_data) >= 20 or data.get("meta", {}).get("is_end"):
                        break
            if dest_data:
                st.session_state.search_status = f"⚡ Kakao Local API (Found {len(dest_data)} matches)"
                return dest_data
        except Exception as e:
            print(f"Kakao Local API error: {e}")
            # If it fails, fall through to the OSM fallback below

    name_synonyms = {
        "식당": "식당|음식점|김밥|파스타|피자|치킨|돈까스|국밥|레스토랑|가든|분식",
        "음식점": "식당|음식점|김밥|파스타|피자|치킨|돈까스|국밥|레스토랑|가든|분식",
        "restaurant": "식당|음식점|김밥|파스타|피자|치킨|돈까스|국밥|레스토랑|가든|분식",
        "맛집": "식당|음식점|김밥|파스타|피자|치킨|돈까스|국밥|레스토랑|카페|베이커리",
        "술집": "술집|호프|맥주|이자카야|포차|바|pub|bar",
        "병원": "병원|의원|내과|외과|소아과|치과|피부과|안과|정형외과",
        "약국": "약국|프라자약국|온누리약국",
        "카페": "카페|커피|디저트|베이커리|다방|cafe|coffee",
        "cafe": "카페|커피|디저트|베이커리|다방|cafe|coffee",
        "편의점": "편의점|CU|GS25|세븐일레븐|이마트24|미니스톱",
        "마트": "마트|이마트|홈플러스|롯데마트|슈퍼"
    }
    
    category_tags = {
        "식당": [("amenity", "restaurant"), ("amenity", "fast_food"), ("amenity", "food_court")],
        "음식점": [("amenity", "restaurant"), ("amenity", "fast_food"), ("amenity", "food_court")],
        "레스토랑": [("amenity", "restaurant")],
        "restaurant": [("amenity", "restaurant"), ("amenity", "fast_food")],
        "맛집": [("amenity", "restaurant"), ("amenity", "fast_food"), ("amenity", "cafe")],
        "카페": [("amenity", "cafe")],
        "cafe": [("amenity", "cafe")],
        "커피": [("amenity", "cafe")],
        "편의점": [("shop", "convenience")],
        "마트": [("shop", "supermarket"), ("shop", "convenience")],
        "술집": [("amenity", "bar"), ("amenity", "pub")],
        "병원": [("amenity", "hospital"), ("amenity", "clinic")],
        "약국": [("amenity", "pharmacy")],
        "공원": [("leisure", "park")]
    }
    
    search_regex = name_synonyms.get(keyword.lower(), keyword)
    
    queries = []
    queries.append(f'node(around:{radius},{avg_lat},{avg_lon})["name"~"{search_regex}"];')
    queries.append(f'way(around:{radius},{avg_lat},{avg_lon})["name"~"{search_regex}"];')
    
    if keyword.lower() in category_tags:
        for k, v in category_tags[keyword.lower()]:
            queries.append(f'node(around:{radius},{avg_lat},{avg_lon})["{k}"="{v}"];')
            queries.append(f'way(around:{radius},{avg_lat},{avg_lon})["{k}"="{v}"];')
            
    query_str = "\n".join(queries)
    overpass_query = f"[out:json];({query_str});out center;"
    
    try:
        resp = requests.get("http://overpass-api.de/api/interpreter", params={'data': overpass_query}, timeout=10)
        if resp.status_code == 200:
            elements = resp.json().get('elements', [])
            seen = set()
            for el in elements:
                tags = el.get('tags', {})
                name = tags.get('name')
                if not name or name in seen:
                    continue
                seen.add(name)
                
                lat = el['lat'] if el['type'] == 'node' else el['center']['lat']
                lon = el['lon'] if el['type'] == 'node' else el['center']['lon']
                
                addr_parts = [tags.get(k, '') for k in ('addr:city', 'addr:street', 'addr:housenumber')]
                addr = " ".join(filter(None, addr_parts)) or (tags.get('amenity') or tags.get('shop') or "OSM POI")
                
                dest_data.append({"name": name, "address": addr, "coords": (lat, lon)})
                if len(dest_data) >= 20:
                    break
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"Overpass API error: {e}")
        
    if dest_data and (not st.session_state.get("search_status") or "Unknown" in st.session_state.search_status or "ONLINE" in st.session_state.search_status):
        if "ONLINE" in st.session_state.get("search_status", ""):
            st.session_state.search_status = f"🤖 SpotSync AI Online (0 DB results) ➔ Fallback: 🌍 OpenStreetMap (Found {len(dest_data)} matches)"
        else:
            st.session_state.search_status = f"🌍 OpenStreetMap (Overpass) (Found {len(dest_data)} matches)"
        
    if len(dest_data) < 20:
        try:
            # Prevent 10,000km global search bug by bounding to the user's starting region in South Korea
            delta = 0.15  # ~15km bounding box
            viewbox = [
                (avg_lat - delta, avg_lon - delta),
                (avg_lat + delta, avg_lon + delta)
            ]
            locations = geolocator.geocode(
                keyword, 
                exactly_one=False, 
                limit=20, 
                viewbox=viewbox, 
                bounded=True, 
                country_codes="kr"
            )
            if locations:
                for loc in locations:
                    short_name = ",".join(loc.address.split(",")[:2])
                    if not any(d["name"] == short_name for d in dest_data):
                        dest_data.append({"name": short_name, "address": loc.address, "coords": (loc.latitude, loc.longitude)})
                        if len(dest_data) >= 20:
                            break
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"Nominatim search error: {e}")
            
    if dest_data and (not st.session_state.get("search_status") or "Unknown" in st.session_state.search_status or "ONLINE" in st.session_state.search_status or "OSM" in st.session_state.search_status):
        if "ONLINE" in st.session_state.get("search_status", ""):
            st.session_state.search_status = f"🤖 SpotSync AI Online (0 DB results) ➔ Fallback: 🗺️ OSM Address Geocoder (Found {len(dest_data)} matches)"
        else:
            st.session_state.search_status = f"🗺️ OSM Address Geocoder Fallback (Found {len(dest_data)} matches)"
            
    return dest_data

# --- Sidebar UI ---
with st.sidebar:
    st.header("1. Starting Points")
    with st.form("origin_form", clear_on_submit=True):
        new_origin = st.text_input("Enter an origin address:")
        submitted = st.form_submit_button("Add Origin")
        if submitted and new_origin:
            with st.spinner("Finding address..."):
                coords = geocode_address(new_origin)
                if coords:
                    st.session_state.origins[new_origin] = coords
                    st.session_state.search_active = False
                else:
                    st.error("Address not found.")
                    
    if st.session_state.origins:
        st.write("### Origin List")
        for orig in list(st.session_state.origins.keys()):
            col1, col2 = st.columns([4, 1])
            col1.write(f"📍 {orig}")
            if col2.button("X", key=f"del_{orig}"):
                del st.session_state.origins[orig]
                st.session_state.search_active = False
                st.rerun()

    st.header("2. Search Destination")
    st.markdown("🤖 **SpotSync AI Local Semantic Search is active!** Enter any natural language search intent:")
    keyword = st.text_input("Search Intent (e.g., '노트북 하기 좋은 카페', '방음 합주실'):")
    similarity_threshold = st.slider("Similarity Threshold (유사도 임계치):", 0.0, 1.0, 0.15, 0.05)
    kakao_api_key = st.text_input("Kakao REST API Key (Optional, for fallback web search):", type="password")
    search_btn = st.button("Calculate & Rank", type="primary", use_container_width=True)

# --- Main Area UI ---
if not st.session_state.origins:
    st.info("👈 Please add at least one starting point from the sidebar to begin.")
    st.stop()

origins_data = [(name, coords) for name, coords in st.session_state.origins.items()]
    
# --- Search Trigger & Routing Logic ---
if search_btn and keyword:
    st.session_state.search_active = True
    
    with st.spinner(f"Searching for '{keyword}' near the center point..."):
        dest_data = search_destinations(keyword, origins_data, similarity_threshold, kakao_api_key)
        
    if not dest_data:
        st.session_state.results = []
        st.error(f"No destinations found for '{keyword}'. Try a different keyword.")
    else:
        temp_results = []
        progress_text = "Calculating routes..."
        my_bar = st.progress(0, text=progress_text)
        
        for idx, dest in enumerate(dest_data):
            my_bar.progress((idx + 1) / len(dest_data), text=f"Routing to {dest['name']}...")
            distances = []
            dest_routes = []
            valid = True
            
            for orig_name, orig_coords in origins_data:
                path, dist = get_route_info(orig_coords, dest["coords"])
                if path:
                    distances.append(dist)
                    dest_routes.append({"name": orig_name, "path": path})
                else:
                    valid = False
                    break
                    
            if valid and distances:
                total_dist = sum(distances)
                fairness = statistics.stdev(distances) if len(distances) > 1 else 0.0
                score = total_dist + fairness
                
                temp_results.append({
                    "name": dest["name"],
                    "address": dest["address"],
                    "coords": dest["coords"],
                    "total_dist": total_dist,
                    "fairness": fairness,
                    "score": score,
                    "routes": dest_routes
                })
        my_bar.empty()
        st.session_state.results = temp_results

# --- Render Decisions ---
if not st.session_state.search_active or not st.session_state.results:
    # Render default map with starting points
    origins_list = list(st.session_state.origins.values())
    avg_lat = sum(c[0] for c in origins_list) / len(origins_list)
    avg_lon = sum(c[1] for c in origins_list) / len(origins_list)
    
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=12)
    for name, coords in st.session_state.origins.items():
        folium.Marker(coords, tooltip=name, icon=folium.Icon(color="blue", icon="user")).add_to(m)
        
    st_folium(m, use_container_width=True, height=600, returned_objects=[])
    st.stop()

# Display Search Engine Status alert
if "search_status" in st.session_state:
    status_str = st.session_state.search_status
    if "🤖" in status_str:
        if "0 DB results" in status_str:
            st.info(f"💡 **안내**: {status_str} (AI DB에서 반경 내 결과를 찾지 못해 일반 주소 검색으로 우회 매칭되었습니다. 아직 벡터 변환이 완료되지 않은 지역일 수 있습니다.)")
        else:
            st.success(f"🔍 **검색 성공**: {status_str}")
    elif "⚡" in status_str:
        st.info(f"💡 **알림 (로컬 AI 백엔드 오프라인)**: {status_str}로 작동 중입니다.")
    else:
        st.warning(f"⚠️ **경고 (로컬 AI 백엔드 & Kakao API 모두 오프라인)**: {status_str}로 구동되었습니다. 자연어 시맨틱 검색 성능이 매우 떨어질 수 있습니다.")

results = st.session_state.results
results.sort(key=lambda x: x["score"])

# --- Display Candidates Table ---
st.subheader("📊 Ranked Candidate Locations Overview")
df_data = []
for i, res in enumerate(results):
    df_data.append({
        "Rank": i + 1,
        "Name": res["name"],
        "Address": res["address"],
        "Total Dist (km)": round(res["total_dist"] / 1000, 2),
        "Fairness Variance (km)": round(res["fairness"] / 1000, 2)
    })

df = pd.DataFrame(df_data).set_index("Rank")
st.dataframe(df, use_container_width=True)
st.divider()

# --- Layout: Map and Results ---
col_map, col_results = st.columns([2, 1])

with col_map:
    st.subheader("Interactive Map")
    top_dest = results[0]
    
    m = folium.Map(location=top_dest["coords"], zoom_start=13)
    
    for i, (orig_name, orig_coords) in enumerate(origins_data):
        folium.Marker(
            orig_coords, 
            tooltip=f"Origin: {orig_name}",
            icon=folium.Icon(color=colors[i % len(colors)], icon="user")
        ).add_to(m)
        
    folium.Marker(
        top_dest["coords"],
        tooltip=f"★ Top Pick: {top_dest['name']}",
        icon=folium.Icon(color="red", icon="star")
    ).add_to(m)
    
    for res in results[1:]:
        folium.Marker(
            res["coords"],
            tooltip=f"Candidate: {res['name']}",
            icon=folium.Icon(color="gray", icon="info-sign")
        ).add_to(m)
        
    for i, route_info in enumerate(top_dest["routes"]):
        folium.PolyLine(
            route_info["path"],
            color=colors[i % len(colors)],
            weight=4,
            opacity=0.8,
            tooltip=f"Route from {route_info['name']}"
        ).add_to(m)
        
    st_folium(m, use_container_width=True, height=600, returned_objects=[])

with col_results:
    st.subheader("Top Ranked Locations")
    for i, res in enumerate(results):
        with st.expander(f"#{i+1}: {res['name']} {'★' if i == 0 else ''}", expanded=(i == 0)):
            st.markdown(f"**Address:** {res['address']}")
            st.markdown(f"**Total Distance:** {res['total_dist']/1000:.2f} km")
            st.markdown(f"**Fairness Variance:** {res['fairness']/1000:.2f} km")