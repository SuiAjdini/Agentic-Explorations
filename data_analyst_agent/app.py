import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import gradio as gr
import google.generativeai as genai

from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.llms.gemini import Gemini
import config

try:
    print("🔑 Configuring Gemini API...")
    genai.configure(api_key=config.GOOGLE_API_KEY)
    print("✅ Gemini API configured.")
except Exception as e:
    print(f"❌ Error configuring Gemini API: {e}")

def create_agent(df: pd.DataFrame):
    llm = Gemini(model_name=f"models/{config.LLM_MODEL_NAME}")
    query_engine = PandasQueryEngine(df=df, llm=llm, verbose=True)
    return query_engine

def process_csv(file):
    if file is None:
        gr.Warning("Please upload a CSV file before loading.")
        return None, None, gr.update(interactive=False)
    try:
        df = pd.read_csv(file.name)
        agent = create_agent(df)
        new_state = {"agent": agent, "df": df}
        return df.head(), new_state, gr.update(interactive=True, placeholder="e.g., Show me a pie chart of my spending by category")
    except Exception as e:
        gr.Error(f"Error processing CSV: {e}")
        return None, None, gr.update(interactive=False)

def chat_with_agent(app_state, user_question):
    plot_output = gr.update(visible=False)
    agent = app_state.get("agent")
    df = app_state.get("df")

    if agent is None or not user_question:
        return "Agent not ready. Please load a CSV and ask a question.", plot_output

    if os.path.exists("plot.png"):
        os.remove("plot.png")

    try:
        response = agent.query(user_question)
        raw_answer = response.response if response.response else ""
        
        # "Detect and Re-run" logic for fixing plots
        if "Axes(" in raw_answer:
            print("🛠️ Plot detected! Re-running code with manual fix...")
            original_code = response.metadata.get('pandas_instruction_str', None)
            if original_code:
                fixed_code = f"import matplotlib.pyplot as plt\n{original_code}\nplt.savefig('plot.png')\nplt.close()"
                print("Executing fixed code block...")
                # NEW: Use the df from our state for the execution context
                local_scope = {'df': df}
                exec(fixed_code, globals(), local_scope)
                print("✅ Chart saved to plot.png")

        # Post-processing to create clean text responses
        final_answer = raw_answer
        if "Axes(" in raw_answer or os.path.exists("plot.png"):
            final_answer = "I have generated the requested chart. It is now displayed in the chart panel."
        elif all(keyword in raw_answer for keyword in ['count', 'mean', 'std', 'unique']):
            try:
                summarizer_model = genai.GenerativeModel(config.LLM_MODEL_NAME)
                prompt = f"Please summarize the following statistical data in a friendly, natural language paragraph:\n\n{raw_answer}"
                summary_response = summarizer_model.generate_content(prompt)
                final_answer = summary_response.text
            except Exception:
                final_answer = "I found some statistics about your data, but couldn't summarize them:\n\n" + raw_answer

        if os.path.exists("plot.png"):
            plot_output = gr.update(value="plot.png", visible=True)

        return final_answer, plot_output
        
    except Exception as e:
        print(f"----------- An error occurred during query: {e} -----------")
        return f"Error during query: {e}", plot_output

# --- Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft(), title="Data Analyst Agent") as app:
    gr.Markdown("# 📈 Data Analyst Agent")
    # ... UI elements 
    app_state = gr.State(value={"agent": None, "df": None})

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload your CSV file", file_types=[".csv"])
            load_btn = gr.Button("Load CSV & Start Agent", variant="secondary")
            gr.Examples(examples=[["data.csv"]], inputs=[file_input], label="Sample Data")
            df_output = gr.Dataframe(label="CSV Preview")
        with gr.Column(scale=2):
            chatbot_messages = gr.Textbox(label="Agent Response", lines=10, interactive=False)
            plot_output = gr.Image(label="Chart", visible=False, type="filepath")
            with gr.Row():
                chat_input = gr.Textbox(label="Your Question", placeholder="Upload and load a CSV to begin...", scale=4, interactive=False)
                submit_btn = gr.Button("Ask", variant="primary", scale=1)

    load_btn.click(
        fn=process_csv,
        inputs=[file_input],
        outputs=[df_output, app_state, chat_input]
    )

    submit_btn.click(
        fn=chat_with_agent,
        inputs=[app_state, chat_input],
        outputs=[chatbot_messages, plot_output]
    )
    
    chat_input.submit(
        fn=chat_with_agent,
        inputs=[app_state, chat_input],
        outputs=[chatbot_messages, plot_output]
    )

if __name__ == "__main__":
    app.launch()