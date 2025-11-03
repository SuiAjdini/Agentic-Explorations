# 🗞️ Daily AI News Digest Agent

This Streamlit application automatically generates a **daily AI & tech news digest** using:
- **Tavily API** → performs deep web search for current news per topic.
- **Google Gemini (Generative AI)** → synthesizes the retrieved articles into a concise, cited morning briefing.

## 🚀 Features
- Multi-topic configurable search (AI, Cloud, Cybersecurity, Startups, etc.)
- Adjustable parameters: time range, tone, language, and summary length.
- Automatically loads `.env` keys from parent directories or environment variables.
- Displays fetched sources with citations and lets you download the final summary as Markdown.

### 1. Setup
- Set API Key**: Make sure you have created a **`.env`** file in the main project folder and added your `GOOGLE_API_KEY` to it. 
- Get and API Key from https://www.tavily.com/

### 2. Create a virtual environment
```bash
# Create the environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Script

```bash
streamlit run app.py
```