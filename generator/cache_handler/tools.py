import os
import sqlite3

from dotenv import load_dotenv

# --- Configuration ---

load_dotenv()

cache_dir_name = os.getenv("CACHE_DIR", "cache")
db_file_name = os.getenv("LLM_CACHE_FILE_NAME", ".langchain.db")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DB_PATH = os.path.join(PROJECT_ROOT, cache_dir_name, db_file_name)

print(f"Using database at: {DB_PATH}")

# --- Default Table and Column Names ---
DEFAULT_TABLE_NAME = "llm_cache_with_timestamp"
PROMPT_COLUMN = "prompt"
RESPONSE_COLUMN = "response"
TIMESTAMP_COLUMN = "last_accessed_at"


# --- Core Functions ---


def get_connection(db_path):
    """Establishes a connection to the SQLite database."""
    if not os.path.exists(db_path):
        print(f"Error: Database file not found at '{db_path}'")
        return None
    return sqlite3.connect(db_path)


def inspect_db(db_path=DB_PATH):
    """Inspects the database to list all tables and their columns."""
    print(f"\n--- Inspecting Database Schema for '{db_path}' ---")
    conn = get_connection(db_path)
    if not conn:
        return
    with conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        if not tables:
            print("\nDatabase is empty or contains no tables.")
            return
        print(f"\nFound {len(tables)} table(s):")
        for table_name_tuple in tables:
            table_name = table_name_tuple[0]
            print(f"\n=== Table: `{table_name}` ===")
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            print("  Columns:")
            for col in columns:
                print(f"    - {col[1]} (Type: {col[2]})")
    print("\n--- Inspection Complete ---")


def list_cache(db_path=DB_PATH, limit=20, table_name=DEFAULT_TABLE_NAME):
    """Lists the prompts, responses, and timestamps in the cache."""
    conn = get_connection(db_path)
    if not conn:
        return
    with conn:
        cursor = conn.cursor()
        query = (
            f"SELECT {PROMPT_COLUMN}, SUBSTR({RESPONSE_COLUMN}, 1, 80), "
            f"{TIMESTAMP_COLUMN} FROM {table_name} ORDER BY {TIMESTAMP_COLUMN} "
            f"DESC LIMIT ?"
        )
        print(
            f"\n--- Listing up to {limit} most recent entries from table "
            f"`{table_name}` ---\n"
        )
        try:
            results = cursor.execute(query, (limit,)).fetchall()
            if not results:
                print(f"Table `{table_name}` is empty.")
                return
            for row in results:
                prompt, response_preview, timestamp = row
                print(f"▶ Last Accessed: {timestamp}")
                print(f"  Prompt: {prompt[:150]}...")
                print(f"  Response Preview: {response_preview}...\n")
        except sqlite3.OperationalError as e:
            print(f"An error occurred: {e}")
            print(
                f"Hint: Does table `{table_name}` exist and have the correct "
                "columns?"
            )


def delete_old_entries(
    days_old=30, db_path=DB_PATH, table_name=DEFAULT_TABLE_NAME
):
    """
    Deletes cache entries that have not been accessed in the last N
    days.
    """
    conn = get_connection(db_path)
    if not conn:
        return
    confirm = input(
        f"Are you sure you want to delete entries from `{table_name}` "
        f"not used in the last {days_old} days? (y/n): "
    )
    if confirm.lower() != "y":
        print("Operation cancelled.")
        return
    with conn:
        cursor = conn.cursor()
        # This query deletes entries with a timestamp OLDER than N days
        # ago.
        query = (
            f"DELETE FROM {table_name} WHERE {TIMESTAMP_COLUMN} <= date('now', "
            f"'-{days_old} days')"
        )
        try:
            cursor.execute(query)
            changes = cursor.rowcount
            conn.commit()
            print(
                f"\nSuccessfully deleted {changes} old entries from "
                f"`{table_name}`."
            )
        except sqlite3.OperationalError as e:
            print(f"An error occurred: {e}")
            print(
                f"Hint: Does the table `{table_name}` have a "
                f"`{TIMESTAMP_COLUMN}` column?"
            )


def delete_recent_entries(
    days=7, db_path=DB_PATH, table_name=DEFAULT_TABLE_NAME
):
    """
    Deletes cache entries that WERE accessed within the last N days.
    """
    conn = get_connection(db_path)
    if not conn:
        return
    confirm = input(
        f"Are you sure you want to delete entries from `{table_name}` "
        f"that were used in the last {days} days? (y/n): "
    )
    if confirm.lower() != "y":
        print("Operation cancelled.")
        return
    with conn:
        cursor = conn.cursor()
        # This query deletes entries with a timestamp MORE RECENT than
        # N days ago.
        query = (
            f"DELETE FROM {table_name} WHERE {TIMESTAMP_COLUMN} >= date('now', "
            f"'-{days} days')"
        )
        try:
            cursor.execute(query)
            changes = cursor.rowcount
            conn.commit()
            print(
                f"\nSuccessfully deleted {changes} recent entries from "
                f"`{table_name}`."
            )
        except sqlite3.OperationalError as e:
            print(f"An error occurred: {e}")
            print(
                f"Hint: Does the table `{table_name}` have a "
                f"`{TIMESTAMP_COLUMN}` column?"
            )


def search_cache(search_term, db_path=DB_PATH, table_name=DEFAULT_TABLE_NAME):
    """Searches for a specific prompt in the cache."""
    conn = get_connection(db_path)
    if not conn:
        return
    with conn:
        cursor = conn.cursor()
        query = (
            f"SELECT {PROMPT_COLUMN}, {RESPONSE_COLUMN} FROM {table_name} "
            f"WHERE {PROMPT_COLUMN} LIKE ?"
        )
        search_pattern = f"%{search_term}%"
        print(
            f"\n--- Searching for prompts containing '{search_term}' in "
            f"`{table_name}` ---\n"
        )
        results = cursor.execute(query, (search_pattern,)).fetchall()
        if not results:
            print("No matching entries found.")
            return
        for prompt, response in results:
            print(f"▶ Prompt: {prompt}")
            print(f"  Cached Response: {response}\n")


def delete_entry(
    prompt_to_delete, db_path=DB_PATH, table_name=DEFAULT_TABLE_NAME
):
    """
    Deletes a specific entry from the cache based on the exact prompt.
    """
    conn = get_connection(db_path)
    if not conn:
        return
    with conn:
        cursor = conn.cursor()
        # CORRECTED: Using the right column name in the WHERE clause
        query = f"DELETE FROM {table_name} WHERE {PROMPT_COLUMN} = ?"
        cursor.execute(query, (prompt_to_delete,))
        changes = cursor.rowcount
        conn.commit()
    if changes > 0:
        print(
            f"\nSuccessfully deleted {changes} entry/entries from "
            f"`{table_name}`."
        )
    else:
        print("\nNo entry found with that exact prompt. No changes were made.")


def clear_cache(db_path=DB_PATH, table_name=DEFAULT_TABLE_NAME):
    """Deletes all entries from a specific cache table."""
    conn = get_connection(db_path)
    if not conn:
        return
    confirm = input(
        f"Are you sure you want to delete ALL entries from table "
        f"`{table_name}`? (y/n): "
    )
    if confirm.lower() != "y":
        print("Operation cancelled.")
        return
    with conn:
        cursor = conn.cursor()
        query = f"DELETE FROM {table_name}"
        cursor.execute(query)
        changes = cursor.rowcount
        conn.commit()
    print(f"\nSuccessfully cleared `{table_name}`. Deleted {changes} entries.")


# ======================================================================
# === CONTROL PANEL: UNCOMMENT THE FUNCTION YOU WANT TO RUN ===
# ======================================================================
if __name__ == "__main__":
    # --- 0. Inspect the database schema (to see your new table) ---
    # inspect_db()

    # --- 1. List the most recently used cache entries ---
    list_cache(limit=10)

    # --- 2. Delete OLD cache entries (for cleanup) ---
    # Deletes entries not accessed in the last 30 days.
    # delete_old_entries(days_old=30)

    # --- 3. Delete RECENT cache entries (for clearing tests, etc.) ---
    # Deletes entries accessed in the last 7 days.
    # delete_recent_entries(days=7)

    # Deletes entries from today.
    # delete_recent_entries(days=0)

    # clear_cache()

    print("\nScript finished.")
