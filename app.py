import os

# 🛡️ THE SAFETY SWITCHES
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

import streamlit as st
import sqlite3
import json
import re
import uuid
import pandas as pd
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool

# ==================== GLOBAL CONFIGURATION ====================
st.set_page_config(page_title="Sovereign Global AI", layout="wide", page_icon="👑")

try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    SERPER_API_KEY = st.secrets["SERPER_API_KEY"]
except KeyError:
    st.error("⚠️ **API Keys Missing**: Add them to Streamlit Secrets.")
    st.stop()

os.environ["SERPER_API_KEY"] = SERPER_API_KEY

# OPEN RESPONSES STANDARD
# Set to 8 RPM to stay under your 10 RPM Google limit
global_llm = LLM(
    model="openai/gemini-1.5-flash", 
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=GOOGLE_API_KEY,
    temperature=0.3,
    max_rpm=8
)

# ==================== DATABASE ENGINE ====================

def init_db():
    conn = sqlite3.connect('global_production.db')
    c = conn.cursor()
    # Fixed the triple-quoted string below
    c.execute('''
        CREATE TABLE IF NOT EXISTS projects 
        (id TEXT PRIMARY KEY, name TEXT, location TEXT, type TEXT, budget REAL, plan TEXT)
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS crew 
        (id INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT, role TEXT, name TEXT, rate REAL)
    ''')
    conn.commit()
    conn.close()

init_db()

# ==================== AI AGENT LOGIC ====================

def run_global_ai(data, directive):
    researcher = Agent(
        role="Global Production Researcher",
        goal=f"Research 2026 production data for {data['location']}",
        backstory="Expert in global logistics and equipment markets.",
        tools=[SerperDevTool()],
        llm=global_llm,
        verbose=True
    )
    
    planner = Agent(
        role="Executive Producer",
        goal="Create a bulletproof production plan and budget",
        backstory="Strategic lead for international media production.",
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
        st.info("No projects yet. Start a 'New Global Project'.")
    else:
        st.dataframe(df, use_container_width=True)
        project_list = df['name'].tolist()
        selected_project_name = st.selectbox("Select Project to View", project_list)
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
    st.subheader("➕ Create a Global Production")
    with st.form("creation_form"):
        name = st.text_input("Project Name")
        loc = st.text_input("Location")
        ptype = st.selectbox("Production Type", ["Commercial", "Documentary", "Feature Film"])
        budget = st.number_input("Total Budget", value=15000)
        directive = st.text_area("What should the AI plan?")
        
        if st.form_submit_button("🚀 Launch Global AI Team"):
            p_id = str(uuid.uuid4())[:8]
            with st.status("🧠 Agents coordinating...") as status:
                result = run_global_ai({'location': loc, 'type': ptype, 'budget': budget}, directive)
                
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
                        status.update(label="✅ Success!", state="complete")
                        st.balloons()
                    else:
                        st.error("AI failed to format data.")
                except Exception as e:
                    st.error(f"Error: {e}")
