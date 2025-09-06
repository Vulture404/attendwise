import sqlite3
import pandas as pd
import streamlit as st

st.title("📊 Student Attendance Dashboard")

conn = sqlite3.connect("attendance.db")
df = pd.read_sql_query("SELECT * FROM attendance", conn)
conn.close()

st.dataframe(df)

if not df.empty:
    summary = df.groupby("student_name").size().reset_index(name="Days Present")
    st.bar_chart(summary.set_index("student_name"))
