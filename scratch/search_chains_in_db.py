import sys
from sqlalchemy import create_engine, text

sys.stdout.reconfigure(encoding='utf-8')
DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/spotsync"

try:
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        print("Checking CGV in DB...")
        cgv_results = conn.execute(text("""
            SELECT id, name, category, address 
            FROM places 
            WHERE name ILIKE '%CGV%' 
            LIMIT 5
        """)).all()
        for r in cgv_results:
            print(f"  - {r[1]} ({r[2]}) | Address: {r[3]}")
            
        print("\nChecking McDonald's in DB...")
        mcd_results = conn.execute(text("""
            SELECT id, name, category, address 
            FROM places 
            WHERE name ILIKE '%맥도날드%' OR name ILIKE '%맥도널드%'
            LIMIT 5
        """)).all()
        for r in mcd_results:
            print(f"  - {r[1]} ({r[2]}) | Address: {r[3]}")
            
        print("\nChecking Starbucks in DB...")
        starbucks_results = conn.execute(text("""
            SELECT id, name, category, address 
            FROM places 
            WHERE name ILIKE '%스타벅스%'
            LIMIT 5
        """)).all()
        for r in starbucks_results:
            print(f"  - {r[1]} ({r[2]}) | Address: {r[3]}")
            
except Exception as e:
    print("Error:", e)
