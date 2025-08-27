import os
import pandas as pd
import matplotlib
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
import gradio as gr
import google.generativeai as genai

from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.llms.gemini import Gemini
import data_analyst_agent.gradio.config as config

# --- Gemini API Configuration ---
try:
    print("🔑 Configuring Gemini API...")
    genai.configure(api_key=config.GOOGLE_API_KEY)
    print("✅ Gemini API configured.")
except Exception as e:
    print(f"❌ Error configuring Gemini API: {e}")

# --- Core Functions ---
def create_agent(df: pd.DataFrame):
    """Creates a PandasQueryEngine agent with the specified DataFrame."""
    llm = Gemini(model_name=f"models/{config.LLM_MODEL_NAME}")
    query_engine = PandasQueryEngine(
        df=df, 
        llm=llm, 
        verbose=True,
        instruction_str="""
            You are a helpful data analysis assistant.
            Given a pandas dataframe `df`, and a conversation history, answer the user's question.
            
            IMPORTANT RULES:
            - Your code response MUST NOT contain any imports, like `import matplotlib.pyplot as plt`.
            - If you are asked to create a plot or chart, you MUST generate the code to save it to 'plot.png'.
            - To do this, chain the save command directly to the plot command. FOR EXAMPLE: df.plot.bar().get_figure().savefig('plot.png')
            - Do not use `plt.show()` or `plt.savefig()`.
        """
    )
    return query_engine

def process_csv(file):
    """Loads a CSV, creates an agent, and updates the UI."""
    if file is None:
        gr.Warning("Please upload a CSV file before loading.")
        return None, None, gr.update(interactive=False), []
    try:
        df = pd.read_csv(file.name)
        agent = create_agent(df)
        new_state = {"agent": agent, "df": df}
        return df.head(), new_state, gr.update(interactive=True, placeholder="e.g., Show me a pie chart of spending by category"), []
    except Exception as e:
        gr.Error(f"Error processing CSV: {e}")
        return None, None, gr.update(interactive=False), []


def chat_with_agent(app_state, user_question, chat_history):
    """Handles all chat interaction, checking for file creation to detect plots."""
    plot_output = gr.update(visible=False)
    download_link = gr.update(visible=False)
    
    agent = app_state.get("agent")
    
    if agent is None or not user_question:
        chat_history.append((user_question, "Agent not ready. Please load a CSV first."))
        return chat_history, plot_output, download_link

    if os.path.exists("plot.png"):
        os.remove("plot.png")

    try:
        formatted_history = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in chat_history])
        query_with_context = f"Conversation history:\n{formatted_history}\n\nUser's new question: {user_question}"
        
        print("Querying agent...")
        response = agent.query(query_with_context)
        raw_answer = str(response)
        print(f"Agent response: {raw_answer}")

        final_answer = ""

        # The most reliable way to check for a plot is to see if the file was created.
        if os.path.exists("plot.png"):
            print("✅ Plot file found.")
            final_answer = "I've generated the requested chart for you."
        
        # If no plot was created, then process the text response.
        else:
            if raw_answer == "None":
                 final_answer = "I've completed the action, but there's no specific data to show."
            elif all(keyword in raw_answer for keyword in ['count', 'mean', 'std', 'unique']):
                try:
                    summarizer_model = genai.GenerativeModel(config.LLM_MODEL_NAME)
                    prompt = f"Please summarize the following statistical data in a friendly, natural language paragraph:\n\n{raw_answer}"
                    summary_response = summarizer_model.generate_content(prompt)
                    final_answer = summary_response.text
                except Exception:
                    final_answer = "I found some statistics about your data, but couldn't summarize them:\n\n" + raw_answer
            else:
                final_answer = raw_answer

        chat_history.append((user_question, final_answer))

        # Update UI if a plot was created
        if os.path.exists("plot.png"):
            plot_output = gr.update(value="plot.png", visible=True)
            download_link = gr.update(visible=True, value="plot.png")

        return chat_history, plot_output, download_link
        
    except Exception as e:
        error_message = f"Sorry, I encountered an error: {e}"
        chat_history.append((user_question, error_message))
        print(f"----------- An error occurred during query: {e} -----------")
        return chat_history, plot_output, download_link

# --- Gradio UI (No changes needed) ---
with gr.Blocks(theme=gr.themes.Soft(), title="Data Analyst Agent") as app:
    gr.Markdown("# 📈 Conversational Data Analyst Agent")
    app_state = gr.State(value={"agent": None, "df": None})

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload your CSV file", file_types=[".csv"])
            load_btn = gr.Button("Load CSV & Start Agent", variant="secondary")
            gr.Examples(examples=[["data.csv"]], inputs=[file_input], label="Sample Data")
            df_output = gr.Dataframe(label="CSV Preview")
            
        with gr.Column(scale=2):
            chatbot_ui = gr.Chatbot(label="Conversation with Agent", height=500)
            
            with gr.Accordion("Chart Output", open=False):
                plot_output = gr.Image(label="Chart", visible=False, type="filepath")
                download_plot_btn = gr.File(label="Download Chart", visible=False)

            with gr.Row():
                chat_input = gr.Textbox(
                    label="Your Question", 
                    placeholder="Upload and load a CSV to begin...", 
                    scale=4, 
                    interactive=False
                )
                submit_btn = gr.Button("Ask", variant="primary", scale=1)

    # --- Event Handlers (No changes needed) ---
    load_btn.click(
        fn=process_csv,
        inputs=[file_input],
        outputs=[df_output, app_state, chat_input, chatbot_ui] 
    )

    def submit_message(app_state, user_question, chat_history):
        updated_history, plot, download = chat_with_agent(app_state, user_question, chat_history)
        return updated_history, plot, download, ""

    submit_btn.click(
        fn=submit_message,
        inputs=[app_state, chat_input, chatbot_ui],
        outputs=[chatbot_ui, plot_output, download_plot_btn, chat_input] 
    )
    
    chat_input.submit(
        fn=submit_message,
        inputs=[app_state, chat_input, chatbot_ui],
        outputs=[chatbot_ui, plot_output, download_plot_btn, chat_input]
    )

if __name__ == "__main__":
    app.launch(debug=True)