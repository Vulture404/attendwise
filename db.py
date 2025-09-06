import sqlite3

def init_db():
    conn = sqlite3.connect("attendance.db")
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT,
            date TEXT,
            time TEXT
        )
        """
    )
    conn.commit()
    conn.close()

def mark_attendance(student_name, date, time):
    conn = sqlite3.connect("attendance.db")
    cur = conn.cursor()
    cur.execute("INSERT INTO attendance (student_name, date, time) VALUES (?, ?, ?)", (student_name, date, time))
    conn.commit()
    conn.close()
