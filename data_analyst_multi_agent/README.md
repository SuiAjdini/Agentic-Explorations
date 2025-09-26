#  Multi-Agent Data Analyst 🤖🤝
A Streamlit app demonstrating a multi-agent AI system for data analysis. Upload a CSV, ask a question, and a team of specialized AI agents will work together to give you answers and visualizations.

## How It Works
This project uses a "Project Manager" agent (powered by Google Gemini) to coordinate a team:

A SQL Analyst writes DuckDB queries to fetch data.

A Visualizer writes Matplotlib code to create plots.

The Streamlit app executes the code from both agents to produce the final analysis and chart.

Tech Stack
Frontend: Streamlit | AI Framework: Agno | LLM: Google Gemini | Data: DuckDB, Pandas | Plotting: Matplotlib

Quickstart
### 1. Create a virtual environment
```bash
# Create the environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Set API Key:

Create a .env file.

### 4. Run the App:

```bash
streamlit run app.py
```


###  Why a Multi-Agent Team?
While a single agent can handle simple tasks, this multi-agent pattern is designed for more complex workflows. By breaking a problem down, each specialist agent has a clearer, simpler task, making the system more robust, scalable, and easier to debug.