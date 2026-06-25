import os
import shutil
import glob

# Mapping rules
moves = []

# Logs and outputs
for f in glob.glob("*.log"): moves.append((f, "logs/"))
for f in glob.glob("*.txt"):
    if f not in ["requirements.txt", "backend_url.txt", "frontend_url.txt", "[필독]파일열람방법.txt"]:
        moves.append((f, "logs/"))

# Old ai_search versions
for f in glob.glob("ai_search_v*.py"):
    if f not in ["ai_search_v9.py"]:
        moves.append((f, "archive/"))
moves.append(("ai_search.py", "archive/"))
moves.append(("code.py", "archive/"))

# Embeddings and large data files
for f in glob.glob("*.npz"): moves.append((f, "data/embeddings/"))
for f in glob.glob("*.pt"): moves.append((f, "data/embeddings/"))
for f in glob.glob("*.parquet"): moves.append((f, "data/backups/"))
for f in glob.glob("*.dump"): moves.append((f, "data/backups/"))
for f in glob.glob("*.zip"): moves.append((f, "data/backups/"))

# Tests and Checks
for f in glob.glob("test_*.py"): moves.append((f, "tests/"))
for f in glob.glob("check_*.py"): moves.append((f, "tests/"))

# Scripts
scripts = [
    "add_col.py", "apply_rich_db.py", "apply_v6_db.py", "async_hybrid_scraper.py", "async_scraper.py",
    "categories.py", "check.py", "collect_places.py", "compute_embeddings.py", "compute_embeddings_v4.py",
    "compute_v4_local.py", "count_reviews.py", "data_pipeline.py", "db_migrate_auth.py",
    "dump_embeddings.py", "dump_embeddings2.py", "dump_embeddings_fast.py", "dump_embeddings_keyset.py",
    "dump_embeddings_offset.py", "embed_existing_blogs.py", "enable_vector.py", "enrich_kakao.py",
    "enrich_naver.py", "enrich_seoul_hybrid.py", "export_all_chunked.py", "export_chunked_for_bge.py",
    "export_mapo_enriched.py", "export_mapo_for_bge.py", "export_v4.py", "export_v6.py", "fast_apply.py",
    "fetch_all_kto.py", "fetch_osm_brands.py", "gcp_main.py", "generate_google_embeddings.py",
    "generate_google_embeddings_async.py", "generate_ner_data.py", "harness_summarizer.py",
    "harness_summarizer_poc.py", "index_es.py", "index_es_google.py", "index_es_rich.py",
    "load_filtered_places.py", "load_filtered_places_fixed.py", "nightly_sync.py", "playwright_scraper.py",
    "precompute_embeddings.py", "reset_vertex.py", "reset_vertex_datastore.py", "run_scheduler.py",
    "scratch_count.py", "search_bling.py", "show_enriched.py", "sweep_hybrid_ratio.py",
    "update_db_summaries.py", "update_tourapi.py", "upload_server.py", "wait_and_update_v6.py"
]

for f in scripts:
    if os.path.exists(f):
        moves.append((f, "scripts/"))

moves.append(("run_daily_scraper.bat", "scripts/"))

for src, dst in moves:
    if os.path.isfile(src):
        try:
            shutil.move(src, os.path.join(dst, os.path.basename(src)))
            print(f"Moved {src} -> {dst}")
        except Exception as e:
            print(f"Failed {src}: {e}")
