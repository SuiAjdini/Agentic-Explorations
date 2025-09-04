import os
import re
import tempfile
from typing import List

from dotenv import load_dotenv
load_dotenv()  # Load variables from .env into os.environ

import duckdb
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import streamlit as st
from agno.agent import Agent
from agno.models.google import Gemini
from agno.team import Team
from agno.tools import Toolkit, tool

# -----------------------------
# Config & Bootstrap
# -----------------------------
st.set_page_config(page_title="Multi-Agent Data Analyst", layout="wide")
st.title("🤝 Multi-Agent Data Analyst")

# Sidebar for API Key
api_key_default = os.getenv("GOOGLE_API_KEY", "")
if not api_key_default and "GOOGLE_API_KEY" in st.secrets:
    api_key_default = st.secrets["GOOGLE_API_KEY"] or ""

api_key_input = st.sidebar.text_input("Google API Key", value=api_key_default, type="password")
api_key = (api_key_input or api_key_default).strip()
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

# -----------------------------
# Data Analysis Toolkit
# -----------------------------
class DataTools(Toolkit):
    """A toolkit for data loading and querying using DuckDB."""
    def __init__(self):
        super().__init__(name="data_tools")
        self.con = duckdb.connect(database=":memory:")

    @staticmethod
    def _sanitize_table_name(path: str) -> str:
        """Creates a safe SQL table name from a file path."""
        base_name = os.path.splitext(os.path.basename(path))[0]
        safe_name = re.sub(r'[^a-zA-Z0_9_]', '_', base_name).lower()
        return f'_{safe_name}' if safe_name and safe_name[0].isdigit() else safe_name

    def _load_csv_internal(self, path: str) -> str:
        table_name = self._sanitize_table_name(path)
        self.con.execute(
            f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM read_csv_auto(?, HEADER=TRUE)',
            [path]
        )
        n = self.con.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
        return f"Loaded {n} rows into table '{table_name}'."

    @tool(show_result=True)
    def list_tables(self) -> str:
        """List available tables in the analytics database."""
        rows = self.con.execute("SELECT table_name FROM duckdb_tables() WHERE database_name='memory' ORDER BY table_name").fetchall()
        names = [r[0] for r in rows]
        return "Tables: " + (", ".join(names) if names else "(none)")

    @tool(show_result=True)
    def describe_table(self, table: str) -> str:
        """Show schema & a few sample rows for a table. Args: table (str)."""
        info = self.con.execute(f"PRAGMA table_info('{table}')").df()
        head = self.con.execute(f'SELECT * FROM "{table}" LIMIT 5').df()
        schema_txt = "\n".join([f"- {r['name']}: {r['type']}" for _, r in info.iterrows()])
        sample_txt = head.to_csv(index=False)
        return f"Schema for {table}:\n{schema_txt}\n\nSample rows (up to 5):\n{sample_txt}"

    @tool(show_result=True)
    def sql(self, query: str, limit: int = 1000) -> str:
        """Run a SQL query. If no LIMIT in query, applies a default limit."""
        q = query if " limit " in query.lower() else f"{query.rstrip(';')} LIMIT {limit}"
        df = self.con.execute(q).df()
        return f"Rows: {len(df)}\n\n{df.to_csv(index=False)}"

# -----------------------------
# Build the Agent Team
# -----------------------------
@st.cache_resource(show_spinner="Building agent team...")
def build_team():
    data_tools = DataTools()

    analyst = Agent(
        name="SQL_Analyst",
        role="Expert at writing SQL queries to answer data questions.",
        model=Gemini(id="models/gemini-2.0-flash"),
        tools=[data_tools],
        instructions=[
            "Given a user's question, your ONLY job is to write the best possible SQL query to get the data needed for the analysis and visualization.",
            "First, use `list_tables()` and `describe_table()` to understand the data.",
            "Then, return ONLY the final SQL query in a SQL code block. For example: ```sql\nSELECT * FROM my_table;\n```",
            "Do NOT return any other text, explanation, or analysis.",
        ],
    )

    visualizer = Agent(
        name="Visualizer",
        role="Expert at creating data visualizations by writing Python code.",
        model=Gemini(id="models/gemini-2.0-flash"),
        tools=[],
        instructions=[
            "Your code response MUST NOT contain any imports, like `import pandas as pd`.",
            "Your response should be ONLY the Python code to generate a plot.",
            "The code should be wrapped in a standard Python code block.",
            "Do NOT use `plt.show()` or `plt.savefig()`.",
            "The data is in a pandas DataFrame called `df`. You have access to `matplotlib.pyplot` as `plt`.",
            "Crucially, all plotting commands MUST draw on the provided axes object, `ax`. For example, use `df.plot(kind='bar', ax=ax)`.",
            "To add a title, use a separate line like `ax.set_title('My Title')`. DO NOT put the title inside the `.plot()` function.",
        ],
    )

    team = Team(
        mode="coordinate",
        members=[analyst, visualizer],
        model=Gemini(id="models/gemini-2.0-flash"),
        tools=[],
        instructions=[
            "You are a project manager for a data analysis team.",
            "Step 1: Delegate to the `SQL_Analyst` to get the correct SQL query based on the user's request.",
            "Step 2: Delegate to the `Visualizer` to get the Python plotting code for that request.",
            "Step 3: Present a final, consolidated response.",
            "Your final response MUST include a textual summary of the findings and BOTH the SQL and Python code blocks.",
        ],
        markdown=True,
        show_members_responses=True,
    )
    return team, data_tools
team, data_tools = build_team()

# -----------------------------
# Sidebar: Data Upload
# -----------------------------
st.sidebar.header("📦 Upload Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV file",
    type=["csv"],
    accept_multiple_files=False,
)

if uploaded_file:
    data_dir = os.path.join(tempfile.gettempdir(), "agno_streamlit_data")
    os.makedirs(data_dir, exist_ok=True)
    dest = os.path.join(data_dir, uploaded_file.name)
    with open(dest, "wb") as out:
        out.write(uploaded_file.getvalue())

    with st.spinner("Loading data..."):
        msg = data_tools._load_csv_internal(path=dest)
        st.sidebar.success(msg)
        st.session_state.history = []

# -----------------------------
# Main Chat UI
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

def display_response_and_execute_code(content: str):
    """
    Parses agent response for text, SQL, and Python code blocks.
    Executes the code to render a plot in Streamlit but does NOT show the code.
    """
    sql_match = re.search(r"```sql\n(.*?)```", content, re.DOTALL)
    python_match = re.search(r"```python\n(.*?)```", content, re.DOTALL)

    plain_text = re.sub(r"```(sql|python)\n.*?```", "", content, flags=re.DOTALL).strip()
    if plain_text:
        st.markdown(plain_text)

    if python_match and sql_match:
        sql_query = sql_match.group(1).strip()
        python_code = python_match.group(1).strip()

        try:
            with st.spinner("Executing query and generating plot..."):
                df = data_tools.con.execute(sql_query).df()

                if df.empty:
                    st.warning("The query returned no data. Cannot generate plot.")
                else:

                    fig, ax = plt.subplots(figsize=(10, 6)) 
                    
                    exec_globals = {"df": df, "plt": plt, "ax": ax}
                    exec(python_code, exec_globals)
                    
                    plt.tight_layout()
                    st.pyplot(fig)

        except Exception as e:
            st.error(f"An error occurred while executing the code:\n```\n{e}\n```")

# Display chat history
for role, content in st.session_state.history:
    with st.chat_message(role):
        if role == "assistant":
            display_response_and_execute_code(content)
        else:
            st.markdown(content)

prompt = st.chat_input("Ask a question about your data...")

if prompt:
    st.session_state.history.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🤖 Agents are working..."):
            reply = team.run(prompt)
            if hasattr(reply, "content"):
                reply = reply.content

            display_response_and_execute_code(reply)

    st.session_state.history.append(("assistant", reply))

st.caption("Built with Agno Teams + Google Gemini")