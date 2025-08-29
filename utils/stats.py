# Utility module for tracking and visualizing user statistics

import sqlite3
import matplotlib.pyplot as plt
import os
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Initialize SQLite database
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'stats.db')


def init_db():
    """Initialize the SQLite database and create the stats table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_stats (
            user_id INTEGER,
            label TEXT,
            count INTEGER,
            PRIMARY KEY (user_id, label)
        )
    ''')
    conn.commit()
    conn.close()
    logger.info("SQLite database initialized at %s", DB_PATH)


# Initialize the database when the module is loaded
init_db()


def update_user_stats(user_id: int, label: str):
    """
    Update the user's classification statistics in the SQLite database.

    Args:
        user_id (int): Telegram user ID.
        label (str): Classification label (Hate Speech, Offensive, Normal).
    """
    logger.debug("Updating stats for user %d with label %s", user_id, label)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Check if the user-label pair exists
    cursor.execute(
        "SELECT count FROM user_stats WHERE user_id = ? AND label = ?",
        (user_id, label)
    )
    result = cursor.fetchone()

    if result:
        # Update existing count
        new_count = result[0] + 1
        cursor.execute(
            "UPDATE user_stats SET count = ? WHERE user_id = ? AND label = ?",
            (new_count, user_id, label)
        )
    else:
        # Insert new record
        cursor.execute(
            "INSERT INTO user_stats (user_id, label, count) VALUES (?, ?, ?)",
            (user_id, label, 1)
        )

    conn.commit()
    conn.close()
    logger.debug("Stats updated for user %d: %s", user_id, label)


def generate_stats_report(user_id: int) -> str:
    """
    Generate a pie chart of the user's classification statistics.

    Args:
        user_id (int): Telegram user ID.

    Returns:
        str: Path to the generated chart image, or None if no stats.
    """
    logger.info("Generating stats report for user %d", user_id)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Fetch stats
    cursor.execute(
        "SELECT label, count FROM user_stats WHERE user_id = ?",
        (user_id,)
    )
    stats = cursor.fetchall()
    conn.close()

    if not stats:
        logger.warning("No stats found for user %d", user_id)
        return None

    # Prepare data for pie chart
    labels = [row[0] for row in stats]
    counts = [row[1] for row in stats]

    # Create pie chart
    plt.figure(figsize=(6, 6))
    plt.pie(counts, labels=labels, autopct='%1.1f%%', colors=['#ff9999', '#66b3ff', '#99ff99'])
    plt.title(f'Classification Stats for User {user_id}')

    # Save the chart
    chart_path = f'stats_{user_id}.png'
    plt.savefig(chart_path)
    plt.close()
    logger.info("Stats chart saved to %s", chart_path)
    return chart_path