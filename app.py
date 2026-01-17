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
    st.error("⚠️ **API Keys Missing**")
    st.stop()

os.environ["SERPER_API_KEY"] = SERPER_API_KEY

# 🌎 REFINED GLOBAL CONNECTION
# This uses the direct Gemini model name to avoid "BadRequest" errors
global_llm = LLM(
    model="gemini/gemini-1.5-flash", 
    api_key=GOOGLE_API_KEY,
    temperature=0.2,
    max_rpm=8
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
    # Agent 1: The Local Scout
    researcher = Agent(
        role="Local Production Scout",
        goal=f"Research 2026 production specifics in {data['location']}",
        backstory="Expert in local vendor pricing, unions, and weather logistics.",
        tools=[SerperDevTool()],
        llm=global_llm,
        verbose=True,
        max_iter=3 # 👈 Limits how many times it can 'loop' to prevent overload
    )
    
    # Agent 2: The Producer
    planner = Agent(
        role="Executive Producer",
        goal="Finalize a JSON production budget and schedule",
        backstory="Strategic lead focused on financial viability and risk management.",
        llm=global_llm,
        verbose=True
    )

    t1 = Task(
        description=f"Find 2026 crew rates and equipment availability in {data['location']} for: {directive}",
        expected_output="A concise report of costs and vendors.",
        agent=researcher
    )
    
    t2 = Task(
        description="Transform research into a JSON plan. Use 'crew' (role, name, rate), 'equipment', and 'schedule' keys.",
        expected_output="Return ONLY a JSON object.",
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
    st.subheader("🌍 Project History")
    conn = sqlite3.connect('global_production.db')
    df = pd.read_sql_query("SELECT id, name, location, budget FROM projects", conn)
    conn.close()
    
    if df.empty:
        st.info("No projects yet.")
    else:
        st.dataframe(df, use_container_width=True)
        selected_name = st.selectbox("Select Project", df['name'])
        selected_id = df[df['name'] == selected_name]['id'].values[0]
        
        if st.button("📊 Open Intelligence Report"):
            conn = sqlite3.connect('global_production.db')
            crew_df = pd.read_sql_query("SELECT role, name, rate FROM crew WHERE project_id=?", conn, params=(selected_id,))
            proj_data = pd.read_sql_query("SELECT plan FROM projects WHERE id=?", conn, params=(selected_id,)).iloc[0]
            conn.close()
            
            st.write(f"### Crew List: {selected_name}")
            st.table(crew_df)
            st.write("### AI Strategic Plan")
            st.info(proj_row['plan'])

elif page == "New Global Project":
    st.subheader("➕ Create a New Production")
    with st.form("creation_form"):
        name = st.text_input("Project Name (e.g. Iceland March Shoot)")
        loc = st.text_input("Location (City, Country)")
        ptype = st.selectbox("Type", ["Commercial", "Documentary", "Feature Film"])
        budget = st.number_input("Budget", value=30000)
        directive = st.text_area("Specific Instructions (e.g. Weather analysis, specific cameras)")
        
        if st.form_submit_button("🚀 Launch AI Production Team"):
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
                        status.update(label="✅ Analysis Complete!", state="complete")
                        st.balloons()
                    else:
                        st.error("AI results received but format was not saved to database.")
                        st.write(result)
                except Exception as e:
                    st.error(f"Error: {e}")
