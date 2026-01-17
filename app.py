import os

# 🛡️ THE SAFETY SWITCHES (Must be at the very top to prevent system errors)
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

import streamlit as st
import sqlite3
import json
import re
import uuid
import pandas as pd
from datetime import datetime
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool

# ==================== GLOBAL CONFIGURATION ====================
st.set_page_config(page_title="Sovereign Global AI", layout="wide", page_icon="👑")

# Securely load keys from Streamlit Secrets
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    SERPER_API_KEY = st.secrets["SERPER_API_KEY"]
except KeyError:
    st.error("⚠️ **API Keys Missing**: Go to Streamlit Settings -> Secrets and add GOOGLE_API_KEY and SERPER_API_KEY.")
    st.stop()

os.environ["SERPER_API_KEY"] = SERPER_API_KEY

# OPEN RESPONSES STANDARD: Universal connection logic
# Set to 8 RPM to stay under your specific Google Cloud quota of 10
global_llm = LLM(
    model="openai/gemini-1.5-flash", 
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=GOOGLE_API_KEY,
    temperature=0.3,
    max_rpm=8  # 👈 This stops the 'RateLimitError' by pacing the AI
)

# ==================== DATABASE ENGINE ====================

def init_db():
    conn = sqlite3.connect('global_production.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS projects 
                 (id TEXT PRIMARY KEY, name TEXT, location TEXT, type
