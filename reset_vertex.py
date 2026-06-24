# -*- coding: utf-8 -*-
import time
import psycopg2
from google.cloud import discoveryengine_v1beta as discoveryengine
from google.api_core.exceptions import AlreadyExists
from google.protobuf import struct_pb2

PROJECT_ID = '88052320203'
LOCATION = 'global'
DATASTORE_ID = 'spotsync-places-datastore'
ENGINE_ID = 'spotsync-search-engine'
DATABASE_URL = 'postgresql://postgres:postgres@localhost:5432/spotsync'
PARENT = 'projects/' + PROJECT_ID + '/locations/' + LOCATION + '/collections/default_collection'
DS_NAME = PARENT + '/dataStores/' + DATASTORE_ID

ds_client = discoveryengine.DataStoreServiceClient()
ds = discoveryengine.DataStore(
    display_name='SpotSync Places',
    industry_vertical=discoveryengine.IndustryVertical.GENERIC,
    content_config=discoveryengine.DataStore.ContentConfig.NO_CONTENT,
    solution_types=[discoveryengine.SolutionType.SOLUTION_TYPE_SEARCH],
)
print('STEP1: create datastore')
try:
    op = ds_client.create_data_store(parent=PARENT, data_store=ds, data_store_id=DATASTORE_ID)
    result = op.result(timeout=120)
    print('Created:', result.name)
except AlreadyExists:
    print('Already exists')

print('STEP2: load db')
conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()
SQL = 'SELECT id, name, category, address, COALESCE(description, chr(32)), latitude, longitude FROM places WHERE latitude IS NOT NULL LIMIT 3000'
cur.execute(SQL)
rows = cur.fetchall()
conn.close()
print('Loaded', len(rows))

print('STEP3: upload')
doc_client = discoveryengine.DocumentServiceClient()
branch = DS_NAME + '/branches/0'
ok = err = 0
for row in rows:
    pid, name, cat, addr, desc, lat, lng = row
    parts = list(filter(None, [str(name or ''), str(cat or ''), str(addr or ''), str(desc or '')[:200]]))
    content = '; '.join(parts)
    sd = struct_pb2.Struct()
    sd.update({'id': pid, 'title': str(name or 'N/A'), 'category': str(cat or ''), 'address': str(addr or ''), 'content': content, 'latitude': float(lat or 0), 'longitude': float(lng or 0)})
    doc = discoveryengine.Document(id=str(pid), struct_data=sd)
    try:
        doc_client.create_document(parent=branch, document=doc, document_id=str(pid))
        ok += 1
    except AlreadyExists:
        ok += 1
    except Exception as e:
        err += 1
        if err <= 5: print('ERR', pid, str(e)[:60])
    if ok > 0 and ok % 300 == 0:
        print(ok, 'uploaded')
        time.sleep(0.5)
print('Done ok=', ok, 'err=', err)

print('STEP4: link engine')
eng_client = discoveryengine.EngineServiceClient()
eng_name = PARENT + '/engines/' + ENGINE_ID
try:
    eng = eng_client.get_engine(name=eng_name)
    print('Current ds:', eng.data_store_ids)
    if DATASTORE_ID not in eng.data_store_ids:
        eng.data_store_ids = [DATASTORE_ID]
        op = eng_client.update_engine(engine=eng)
        op.result(timeout=60)
        print('Updated')
    else:
        print('Already linked')
except Exception as e:
    print('Engine err:', e)

print('ALL DONE')