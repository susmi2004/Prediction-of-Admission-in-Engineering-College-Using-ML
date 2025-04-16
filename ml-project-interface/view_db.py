import sqlite3

def view_database():
    conn = sqlite3.connect('user_database.db')
    conn.row_factory = sqlite3.Row
    
    # Get all table names
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    # Print each table and its contents
    for table in tables:
        table_name = table[0]
        print(f"\n=== TABLE: {table_name} ===")
        
        # Get column names
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [column[1] for column in cursor.fetchall()]
        print(" | ".join(columns))
        print("-" * 80)
        
        # Get data
        cursor.execute(f"SELECT * FROM {table_name}")
        rows = cursor.fetchall()
        for row in rows:
            # Convert row to dict for easier access
            row_dict = dict(row)
            # Hide password if it exists
            if 'password' in row_dict:
                row_dict['password'] = '********'
            print(" | ".join(str(row_dict.get(col, '')) for col in columns))
    
    conn.close()

if __name__ == "__main__":
    view_database()
