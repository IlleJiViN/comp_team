import psycopg2

def check():
    conn = psycopg2.connect('postgresql://postgres:postgres@localhost:5432/spotsync')
    cur = conn.cursor()
    cur.execute("SELECT SUM(jsonb_array_length(blog_metadata)) FROM places WHERE blog_metadata IS NOT NULL AND jsonb_typeof(blog_metadata) = 'array'")
    total_reviews = cur.fetchone()[0]
    print(f"Total reviews: {total_reviews}")

if __name__ == "__main__":
    check()
