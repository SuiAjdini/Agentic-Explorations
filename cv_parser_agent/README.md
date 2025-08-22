# CV Parser with Gemini & LlamaIndex 📄🤖

This project is a command-line tool designed to streamline the process of extracting key information from resumes. It uses Google's Gemini model via the LlamaIndex framework to parse CV documents and convert their contents into a structured JSON format. 


---

## Features

* **Human-in-the-Loop CLI:** After an initial extraction, an interactive prompt allows you to:
    * `confirm` that the data is 100% correct.
    * `query` the document for clarification (e.g., "What was the start date for their last role?").
    * `correct` any errors by providing the accurate information.

* **Retrieval-Augmented Correction:** This is not "learning" in the sense of retraining the model. When you provide a correction, the tool adds the human-verified data back into its LlamaIndex knowledge base. 

---

## 📂 Project Structure

The project is organized as below:

```plaintext
cv_parser_agent/
├── cv_data/
│   └── example_cv.pdf         # Place input CVs here
│
├── cv_parser/
│   ├── __init__.py            # Makes this a Python package
│   ├── config.py              # Settings and API keys
│   ├── models.py              # All Pydantic data models
│   └── extractor.py           # Core class for LLM interactions
│
├── main_cli.py                # The main script to run the application on terminal
├── app.py                     # The main script to explore the application on web browser using Gradio UI
├── requirements.txt           # Project dependencies
├── .env.example               # Example environment file
├── .gitignore                 # Files to be ignored by Git
└── README.md                  # This file
```
## 🚀 How to Run the Project

Follow these steps to get the application running.

### 1. Prepare Your Files
* **Add CVs**: Place one or more resume files (e.g., `.pdf`, `.docx`) into the **`cv_data`** folder.
* **Set API Key**: Make sure you have created a **`.env`** file in the main project folder and added your `GOOGLE_API_KEY` to it.

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
With your virtual environment activated and dependencies installed, run the app script from your terminal and explore the appplication on http://127.0.0.1:7860.

```bash
python app.py
```