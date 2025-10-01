# 📬 AI Email & Meeting Agent

This Streamlit app lets you describe email and meeting tasks in natural language.  
A Gemini model parses your intent into structured JSON, which is then executed via Gmail and Google Calendar APIs.  

---

## 🚀 Features
- **Natural Language → Structured Action**
  - "Email Alex the weekly update, cc Nina, subject 'Status', body 'All green.'"
  - "Schedule a 45-min sync with alex@example.com tomorrow at 10:00 titled 'Latency Deep-Dive'."
- **Two Actions Supported**
  - Send Email
  - Create Calendar Meeting (with optional Google Meet link)
- **Dry-Run Mode**
  - Preview what would be sent/created without executing.

### 1. Setup
- Set API Key**: Make sure you have created a **`.env`** file in the main project folder and added your `GOOGLE_API_KEY` to it. 
- Create Google APIs (OAuth) Client ID , download the credentials file and save as credentials.json

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