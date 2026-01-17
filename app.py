import os
# 🛡️ THE SAFETY SWITCHES
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"

import streamlit as st
import sqlite3
import json
import re
import uuid
import pandas as pd
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from langchain_google_genai import ChatGoogleGenerativeAI

# ==================== GLOBAL CONFIGURATION ====================
st.set_page_config(page_title="Sovereign Global AI", layout="wide", page_icon="👑")

try:
    # 🔑 Secure Key Loading
    API_KEY = st.secrets["GOOGLE_API_KEY"]
    os.environ["SERPER_API_KEY"] = st.secrets["SERPER_API_KEY"]
except KeyError:
    st.error("⚠️ API Keys Missing in Secrets.")
    st.stop()

# 🌎 THE STABLE BRIDGE (LangChain Version)
# This prevents the 'ClientError 400' by using a more robust message engine
global_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=API_KEY,
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
        role="Local Production Scout",
        goal=f"Research 2026 production specifics in {data['location']}",
        backstory="Expert in local vendor pricing, logistics, and fixing.",
        tools=[SerperDevTool()],
        llm=global_llm,
        verbose=True,
        allow_delegation=False
    )
    
    planner = Agent(
        role="Executive Producer",
        goal="Create a professional JSON production budget and risk report",
        backstory="Strategic lead focused on financial viability and weather risks.",
        llm=global_llm,
        verbose=True,
        allow_delegation=False
    )

    t1 = Task(
        description=f"Research 2026 rates and weather risks in {data['location']} for: {directive}",
        expected_output="A detailed report of costs, vendors, and weather risks.",
        agent=researcher
    )
    
    t2 = Task(
        description="Transform research into a JSON plan. Use ONLY these keys: 'crew' (list of objects with role, name, rate), 'equipment', 'schedule'.",
        expected_output="Return ONLY a clean JSON object.",
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
        selected_name = st.selectbox("Select Project to View", df['name'])
        selected_id = df[df['name'] == selected_name]['id'].values[0]
        
        if st.button("📊 Open Intelligence Report"):
            conn = sqlite3.connect('global_production.db')
            crew_df = pd.read_sql_query("SELECT role, name, rate FROM crew WHERE project_id=?", conn, params=(selected_id,))
            proj_data = pd.read_sql_query("SELECT plan FROM projects WHERE id=?", conn, params=(selected_id,)).iloc[0][0]
            conn.close()
            
            st.write(f"### Crew Manifest: {selected_name}")
            st.table(crew_df)
            st.write("### AI Strategic Strategy & Risk Report")
            st.info(proj_data)

elif page == "New Global Project":
    st.subheader("➕ Create a New Production")
    with st.form("creation_form"):
        name = st.text_input("Project Name (e.g. Iceland March Shoot)")
        loc = st.text_input("Location (City, Country)")
        ptype = st.selectbox("Type", ["Commercial", "Documentary", "Feature Film"])
        budget = st.number_input("Budget (€)", value=30000)
        directive = st.text_area("Detailed Instructions (Paste Iceland text here)")
        
        if st.form_submit_button("🚀 Launch AI Production Team"):
            p_id = str(uuid.uuid4())[:8]
            with st.status("🧠 Agents coordinating across global markets...") as status:
                try:
                    result = run_global_ai({'location': loc, 'type': ptype, 'budget': budget}, directive)
                    
                    # Robust JSON Extraction
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
                        st.error("AI results received but formatting was invalid.")
                        st.write(result)
                except Exception as e:
                    st.error(f"Error: {e}")
