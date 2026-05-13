# database.py - pgvector connection helper
import psycopg2
from pgvector.psycopg2 import register_vector


def get_connection():
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        dbname="rag",
        user="rag",
        password="rag",
    )
    register_vector(conn)
    return conn
