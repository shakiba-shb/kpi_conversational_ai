
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.schema import SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.utilities.sql_database import SQLDatabase
from langchain.tools.sql_database.tool import QuerySQLDataBaseTool, InfoSQLDatabaseTool
from langchain.tools import StructuredTool
from tools.anomaly_tools import detect_anomalies, plot_kpi_anomalies
import os
from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st

# Load SQL database
db = SQLDatabase.from_uri("sqlite:///data/KPI_Anomaly_Detection/data/kpi_dataset.db")

# Register LangChain tools
tools = [
    StructuredTool.from_function(detect_anomalies),
    StructuredTool.from_function(plot_kpi_anomalies),
    QuerySQLDataBaseTool(db=db),
    InfoSQLDatabaseTool(db=db)
]

instructions = (
    """
    You are a helpful assistant for analyzing telecom network KPI data. 
    The data is stored in a SQL database with a single table called 'kpi_data'. 
    Each row represents a daily measurement from a cellular network sector. 
    The table has the following columns: 
        - Date: Date of measurement (YYYY-MM-DD)
        - Site_ID: ID of the network site (e.g., 'SITE_001')
        - Sector_ID: ID of the sector (e.g., 'SITE_001_SECTOR_A')
        - RSRP, SINR, DL_Throughput, UL_Throughput, RTT, Packet_Loss, Call_Drop_Rate, CPU_Utilization, Active_Users, Handover_Success_Rate: These are the KPIs.
    Use the SQL tools for general data questions (e.g., averages, maxima) and use the anomaly detection tools only when users mention anomalies or outliers.
    """
)

load_dotenv()
client = OpenAI(api_key=os.getenv('OPEN_AI_KEY'))
llm = ChatOpenAI(model="gpt-4o")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Prompt structure
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=instructions),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create agent and executor
agent = create_openai_functions_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory
)

# Streamlit app
st.set_page_config(page_title="KPI Chat Agent", page_icon="📊")
st.title("📡 Telecom KPI Chat Agent")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Ask me about anomalies or KPI stats...")
if user_input:
    with st.spinner("Thinking..."):
        result = agent_executor.invoke({"input": user_input})
        response = result.get("output", str(result))

        st.session_state.chat_history.append(("You", user_input))

        # Check if the response is a path to a plot file
        if isinstance(response, str) and response.endswith(".png") and os.path.exists(response):
            st.session_state.chat_history.append(("Agent", "Here is the anomaly plot:"))
            st.chat_message("Agent").image(response)
        else:
            st.session_state.chat_history.append(("Agent", response))

# Display history
for speaker, msg in st.session_state.chat_history:
    st.chat_message(speaker).write(msg)
