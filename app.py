import streamlit as st
import sqlite3
import json
import os
import re
import uuid
import pandas as pd
from datetime import datetime
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool

# ==================== GLOBAL CONFIGURATION ====================
st.set_page_config(page_title="Sovereign Global AI", layout="wide", page_icon="👑")

# Search for keys in Streamlit Secrets
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    SERPER_API_KEY = st.secrets["SERPER_API_KEY"]
except:
    st.error("⚠️ API Keys Missing. Please add GOOGLE_API_KEY and SERPER_API_KEY to Streamlit Secrets.")
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
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT, role TEXT, name TEXT, rate REAL, currency TEXT)''')
    conn.commit()
    conn.close()

init_db()

# ==================== AI AGENT LOGIC ====================

def run_global_ai(data, directive):
    researcher = Agent(
        role="Global Production Researcher",
        goal=f"Research 2026 production data for {data['location']}",
        backstory="Expert in global logistics, local labor laws, and equipment rental markets.",
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
        description=f"Research local crew rates, equipment vendors, and permit rules in {data['location']} for a {data['type']}.",
        expected_output="A list of 2026 rates and local vendor names.",
        agent=researcher
    )
    
    t2 = Task(
        description=f"Based on research and this directive: {directive}, create a JSON plan. Budget: {data['budget']}.",
        expected_output="JSON with keys: 'crew' (list of: role, name, rate), 'equipment', and 'schedule'.",
        agent=planner,
        context=[t1]
    )

    crew = Crew(agents=[researcher, planner], tasks=[t1, t2])
    return crew.kickoff()

def run_stress_test(budget, crew_list):
    auditor = Agent(role="Risk Auditor", goal="Stress test budgets", backstory="Volatility expert.", llm=global_llm)
    t = Task(
        description=f"Test this budget of {budget} against 20% inflation and a 2-day delay. Crew: {crew_list}",
        expected_output="JSON with 'viability_score' (0-100) and 'risk_notes'.",
        agent=auditor
    )
    return str(Crew(agents=[auditor], tasks=[t]).kickoff())

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
        selected_id = st.selectbox("View Details for Project ID", df['id'])
        
        if st.button("📊 Load Project Intelligence"):
            conn = sqlite3.connect('global_production.db')
            crew_df = pd.read_sql_query("SELECT role, name, rate FROM crew WHERE project_id=?", conn, params=(selected_id,))
            proj_data = pd.read_sql_query("SELECT plan, budget FROM projects WHERE id=?", conn, params=(selected_id,)).iloc[0]
            conn.close()
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("### Crew Manifest")
                st.table(crew_df)
            with col2:
                st.write("### 🛡️ AI Stress Test")
                if st.button("Execute Sensitivity Analysis"):
                    with st.spinner("Calculating risks..."):
                        test_res = run_stress_test(proj_data['budget'], crew_df.to_dict('records'))
                        st.json(test_res)

elif page == "New Global Project":
    st.subheader("➕ Create a Territory-Agnostic Production")
    with st.form("creation_form"):
        name = st.text_input("Project Name (e.g. Alpha Global Shoot)")
        loc = st.text_input("Territory / City (Anywhere in the world)")
        ptype = st.selectbox("Production Type", ["Commercial", "Documentary", "Live Broadcast", "Feature Film"])
        budget = st.number_input("Total Budget (€/USD)", value=10000)
        directive = st.text_area("What do you want the AI to plan?")
        
        if st.form_submit_button("🚀 Launch Global AI Team"):
            p_id = str(uuid.uuid4())[:8]
            with st.status("🧠 Agents coordinating across global markets...") as status:
                result = run_global_ai({'location': loc, 'type': ptype, 'budget': budget}, directive)
                
                # Extract JSON
                try:
                    raw_text = str(result)
                    match = re.search(r'\{.*\}', raw_text, re.DOTALL)
                    clean_data = json.loads(match.group())
                    
                    conn = sqlite3.connect('global_production.db')
                    c = conn.cursor()
                    c.execute("INSERT INTO projects VALUES (?,?,?,?,?,?)", (p_id, name, loc, ptype, budget, raw_text))
                    for person in clean_data.get('crew', []):
                        c.execute("INSERT INTO crew (project_id, role, name, rate) VALUES (?,?,?,?)", 
                                 (p_id, person.get('role'), person.get('name'), person.get('rate', 0)))
                    conn.commit()
                    conn.close()
                    status.update(label="✅ Success! Data synced to Global Database.", state="complete")
                    st.balloons()
                except Exception as e:
                    st.error(f"Error parsing AI response: {e}")