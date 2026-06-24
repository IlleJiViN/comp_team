import psycopg2

conn = psycopg2.connect('postgresql://postgres:postgres@localhost:5432/spotsync')
cur = conn.cursor()

# Create users table
cur.execute("""
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    provider VARCHAR(50) NOT NULL,
    provider_id VARCHAR(255) NOT NULL,
    name VARCHAR(100),
    email VARCHAR(255),
    profile_image TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(provider, provider_id)
);
""")

# Create rooms table
cur.execute("""
CREATE TABLE IF NOT EXISTS rooms (
    id VARCHAR(20) PRIMARY KEY, -- short invite code e.g. A1B2C
    host_id INTEGER REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")

# Create room_members table
cur.execute("""
CREATE TABLE IF NOT EXISTS room_members (
    room_id VARCHAR(20) REFERENCES rooms(id) ON DELETE CASCADE,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (room_id, user_id)
);
""")

conn.commit()
conn.close()
print("Migration successful: users, rooms, room_members tables created.")
