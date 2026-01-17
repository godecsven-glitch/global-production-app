import os

# 🛡️ THE SAFETY SWITCHES (Must be at the very top)
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
# This uses your €240 credits via the Google OpenAI-compatible endpoint
global_llm = LLM(
    model="openai/gemini-2.0-flash-exp", 
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=GOOGLE_API_KEY,
    temperature=0.3
)

# ==================== DATABASE ENGINE ====================

def init_db():
    conn = sqlite3.connect('global_production.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS projects 
                 (id TEXT PRIMARY KEY, name TEXT, location TEXT, type TEXT, budget REAL, plan TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS crew 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT, role TEXT, name TEXT, rate REAL)''')
    conn.commit()
    conn.close()

init_db()

# ==================== AI AGENT LOGIC ====================

def run_global_ai(data, directive):
    researcher = Agent(
        role="Global Production Researcher",
        goal=f"Research 2026 production data for {data['location']}",
        backstory="Expert in global logistics, labor laws, and equipment rental markets.",
        tools=[SerperDevTool()],
        llm=global_llm,
        verbose=True
    )
    
    planner = Agent(
        role="Executive Producer",
        goal="Create a bulletproof production plan and budget",
        backstory="Strategic lead with 25 years in international media production.",
        llm=global_llm,
        verbose=True
    )

    t1 = Task(
        description=f"Research 2026 crew rates and gear vendors in {data['location']} for a {data['type']}.",
        expected_output="A list of 2026 rates and local vendor names.",
        agent=researcher
    )
    
    t2 = Task(
        description=f"Based on research and this directive: {directive}, create a JSON plan. Budget: {data['budget']}.",
        expected_output="Return ONLY a JSON object with keys: 'crew' (list of: role, name, rate), 'equipment', and 'schedule'.",
        agent=planner,
        context=[t1]
    )

    crew = Crew(agents=[researcher, planner], tasks=[t1, t2])
    return str(crew.kickoff())

# ==================== USER INTERFACE ====================

st.title("👑 SOVEREIGN GLOBAL PRODUCTION")
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "New Global Project"])

if page == "Dashboard":
    st.subheader("🌍 Your Global Productions")
    conn = sqlite3.connect('global_production.db')
    df = pd.read_sql_query("SELECT id, name, location, budget FROM projects", conn)
    conn.close()
    
    if df.empty:
        st.info("No projects yet. Click 'New Global Project' to start.")
    else:
        st.dataframe(df, use_container_width=True)
        
        # Details view
        selected_project_name = st.selectbox("Select Project to View", df['name'])
        selected_id = df[df['name'] == selected_project_name]['id'].values[0]
        
        if st.button("📊 View Project Intelligence"):
            conn = sqlite3.connect('global_production.db')
            crew_df = pd.read_sql_query("SELECT role, name, rate FROM crew WHERE project_id=?", conn, params=(selected_id,))
            proj_row = pd.read_sql_query("SELECT plan FROM projects WHERE id=?", conn, params=(selected_id,)).iloc[0]
            conn.close()
            
            st.write(f"### Crew Manifest: {selected_project_name}")
            st.table(crew_df)
            with st.expander("View Full AI Strategy"):
                st.write(proj_row['plan'])

elif page == "New Global Project":
    st.subheader("➕ Create a Territory-Agnostic Production")
    with st.form("creation_form"):
        name = st.text_input("Project Name", placeholder="Global Brand Launch")
        loc = st.text_input("Location", placeholder="London, Tokyo, New York...")
        ptype = st.selectbox("Production Type", ["Commercial", "Documentary", "Feature Film"])
        budget = st.number_input("Total Budget (€/USD)", value=15000)
        directive = st.text_area("What should the AI plan?", placeholder="A 2-day cinematic shoot...")
        
        if st.form_submit_button("🚀 Launch Global AI Team"):
            p_id = str(uuid.uuid4())[:8]
            with st.status("🧠 Agents coordinating across global markets...") as status:
                result = run_global_ai({'location': loc, 'type': ptype, 'budget': budget}, directive)
                
                # Robust JSON Extraction
                try:
                    match = re.search(r'\{.*\}', result, re.DOTALL)
                    if match:
                        clean_data = json.loads(match.group())
                        
                        with sqlite3.connect('global_production.db') as conn:
                            c = conn.cursor()
                            c.execute("INSERT INTO projects VALUES (?,?,?,?,?,?)", (p_id, name, loc, ptype, budget, result))
                            for person in clean_data.get('crew', []):
                                c.execute("INSERT INTO crew (project_id, role, name, rate) VALUES (?,?,?,?)", 
                                         (p_id, person.get('role'), person.get('name'), person.get('rate', 0)))
                            conn.commit()
                        
                        status.update(label="✅ Success! Data synced to Global Database.", state="complete")
                        st.balloons()
                        st.success("Project Saved! Go to Dashboard to view.")
                    else:
                        st.error("AI produced a plan but failed to format the data for the database.")
                        st.write(result)
                except Exception as e:
                    st.error(f"Error saving data: {e}")
