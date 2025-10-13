# 🛍️ Smart Shopper AI Agent

An AI-powered product research assistant that searches the web, compares reviews, and recommends the best products — complete with shop links, prices, pros, and cons.

---

## 🚀 Features
1. Search the web for reviews and shop listings

2. Identify top 2–3 contenders

3. Summarize specs, pros/cons, and add direct shop links

4. Provide a clear final recommendation

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