from sqlalchemy import create_engine, text

DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/spotsync"

try:
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        # Fetch 5 sample rows from places table
        sample = conn.execute(text("SELECT id, place_id, name, category, address FROM places LIMIT 5")).all()
        for i, row in enumerate(sample, 1):
            print(f"Row #{i}:")
            print(f"  - id: {row[0]}")
            print(f"  - place_id: {row[1]}")
            print(f"  - name (repr): {repr(row[2])}")
            print(f"  - category (repr): {repr(row[3])}")
            print(f"  - address (repr): {repr(row[4])}")
            print()
except Exception as e:
    print("Error:", e)
