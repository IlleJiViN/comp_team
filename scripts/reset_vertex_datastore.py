"""
Vertex AI Search 데이터스토어 초기화 및 재업로드 스크립트
- spotsync-places-datastore 새로 생성
- PostgreSQL에서 장소 5000개 가져와 업로드
- 기존 검색 엔진에 새 datastore 연결
"""
import os, json, time
import psycopg2
from google.cloud import discoveryengine_v1beta as discoveryengine
from google.api_core.exceptions import AlreadyExists
from google.protobuf import struct_pb2

PROJECT_ID = "88052320203"
LOCATION = "global"
DATASTORE_ID = "spotsync-places-datastore"
ENGINE_ID = "spotsync-search-engine"
DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/spotsync"
DATASTORE_NAME = f"projects/{PROJECT_ID}/locations/{LOCATION}/collections/default_collection/dataStores/{DATASTORE_ID}"

def step1_create_datastore():
    print("\n[STEP 1] Creating datastore...")
    client = discoveryengine.DataStoreServiceClient()
    ds = discoveryengine.DataStore(
        display_name="SpotSync Places",
        industry_vertical=discoveryengine.IndustryVertical.GENERIC,
        content_config=discoveryengine.DataStore.ContentConfig.NO_CONTENT,
        solution_types=[discoveryengine.SolutionType.SOLUTION_TYPE_SEARCH],
    )
    try:
        op = client.create_data_store(
            parent=f"projects/{PROJECT_ID}/locations/{LOCATION}/collections/default_collection",
            data_store=ds,
            data_store_id=DATASTORE_ID,
        )
        result = op.result(timeout=120)
        print(f"  OK: {result.name}")
    except AlreadyExists:
        print("  Already exists, skipping creation")

def step2_update_schema():
    print("\n[STEP 2] Updating schema...")
    client = discoveryengine.SchemaServiceClient()
    schema_json = json.dumps({
        "type": "object",
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "properties": {
            "id":        {"type": "integer", "retrievable": True},
            "title":     {"type": "string", "keyPropertyMapping": "title", "retrievable": True, "searchable": True},
            "category":  {"type": "string", "indexable": True, "searchable": True, "retrievable": True, "dynamicFacetable": True},
            "address":   {"type": "string", "searchable": True, "retrievable": True, "indexable": True},
            "content":   {"type": "string", "searchable": True, "retrievable": True, "indexable": True},
            "latitude":  {"type": "number", "retrievable": True},
            "longitude": {"type": "number", "retrievable": True},
        }
    })
    schema = discoveryengine.Schema(
        name=f"{DATASTORE_NAME}/schemas/default_schema",
        json_schema=schema_json,
    )
    try:
        op = client.update_schema(schema=schema)
        op.result(timeout=60)
        print("  OK: schema updated")
    except Exception as e:
        print(f"  WARN: {e}")

def step3_upload_documents(limit=5000):
    print(f"\n[STEP 3] Loading {limit} places from PostgreSQL...")
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute("""
        SELECT p.id, p.name, p.category, p.address,
               COALESCE(p.description, '') as desc,
               p.latitude, p.longitude
        FROM places p
        WHERE p.latitude IS NOT NULL AND p.longitude IS NOT NULL
        ORDER BY p.id LIMIT %s
    """, (limit,))
    rows = cur.fetchall()
    conn.close()
    print(f"  Loaded {len(rows)} rows")

    client = discoveryengine.DocumentServiceClient()
    parent = f"{DATASTORE_NAME}/branches/0"
    total, errors = 0, 0

    for i, row in enumerate(rows):
        pid, name, category, address, desc, lat, lng = row
        content = "\n".join(filter(None, [
            f"장소명: {name}" if name else "",
            f"카테고리: {category}" if category else "",
            f"주소: {address}" if address else "",
            f"설명: {desc[:300]}" if desc and len(desc) > 5 else "",
        ]))
        sd = struct_pb2.Struct()
        sd.update({
            "id": pid,
            "title": name or "이름없음",
            "category": category or "",
            "address": address or "",
            "content": content,
            "latitude": float(lat) if lat else 0.0,
            "longitude": float(lng) if lng else 0.0,
        })
        doc = discoveryengine.Document(id=str(pid), struct_data=sd)
        try:
            client.create_document(parent=parent, document=doc, document_id=str(pid))
            total += 1
        except AlreadyExists:
            try:
                client.update_document(document=discoveryengine.Document(
                    name=f"{parent}/documents/{pid}", struct_data=sd))
                total += 1
            except Exception as e2:
                errors += 1
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  WARN doc {pid}: {str(e)[:80]}")

        if (i + 1) % 200 == 0:
            print(f"  Progress: {i+1}/{len(rows)} - ok={total} err={errors}")
        if (i + 1) % 50 == 0:
            time.sleep(0.3)

    print(f"  DONE: {total} uploaded, {errors} errors")

def step4_link_engine():
    print("\n[STEP 4] Linking datastore to engine...")
    client = discoveryengine.EngineServiceClient()
    engine_name = f"projects/{PROJECT_ID}/locations/{LOCATION}/collections/default_collection/engines/{ENGINE_ID}"
    try:
        engine = client.get_engine(name=engine_name)
        print(f"  Current datastores: {engine.data_store_ids}")
        if DATASTORE_ID not in engine.data_store_ids:
            engine.data_store_ids = [DATASTORE_ID]
            op = client.update_engine(engine=engine)
            result = op.result(timeout=60)
            print(f"  Updated to: {result.data_store_ids}")
        else:
            print("  Already linked")
    except Exception as e:
        print(f"  WARN: {e}")

def step5_test():
    print("\n[STEP 5] Test search (waiting 10s for indexing)...")
    time.sleep(10)
    client = discoveryengine.SearchServiceClient()
    sc = f"projects/{PROJECT_ID}/locations/{LOCATION}/collections/default_collection/engines/{ENGINE_ID}/servingConfigs/default_search"
    for q in ["카페", "이자카야", "분위기 좋은 식당"]:
        req = discoveryengine.SearchRequest(serving_config=sc, query=q, page_size=3)
        resp = client.search(req)
        res = list(resp.results)
        print(f"  '{q}' -> {len(res)} results")
        for r in res[:2]:
            print(f"    - {dict(r.document.struct_data).get('title', '?')}")

if __name__ == "__main__":
    print("=" * 60)
    print("Vertex AI Search 재구성 시작")
    print("=" * 60)
    step1_create_datastore()
    step2_update_schema()
    step3_upload_documents(limit=5000)
    step4_link_engine()
    step5_test()
    print("\nDone! Restart ai_search_vertex.py after this.")
