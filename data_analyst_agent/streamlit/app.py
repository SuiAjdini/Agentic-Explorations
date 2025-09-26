import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st
import google.generativeai as genai
import ast # Used to safely evaluate the string list of questions
import io # To capture stdout
from contextlib import redirect_stdout # To capture stdout

from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.llms.gemini import Gemini
import config

# --- Gemini API Configuration ---
try:
    genai.configure(api_key=config.GOOGLE_API_KEY)
except AttributeError:
    st.warning("Please set your GOOGLE_API_KEY in a `config.py` file.", icon="⚠️")


# --- Core Functions ---
@st.cache_resource
def create_agent(df: pd.DataFrame):
    """Creates a PandasQueryEngine agent with the specified DataFrame."""
    llm = Gemini(model_name=f"models/{config.LLM_MODEL_NAME}")
    query_engine = PandasQueryEngine(
        df=df,
        llm=llm,
        verbose=True, # verbose=True is what prints to stdout, which we will capture
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

@st.cache_data
def generate_suggested_questions(df: pd.DataFrame):
    """Generates a few questions based on the dataframe's columns using an LLM."""
    model = genai.GenerativeModel(config.LLM_MODEL_NAME)
    # Taking top 15 columns to keep the prompt concise
    column_names = ", ".join(df.columns[:15])
    prompt = f"""
    Given a dataset with the following columns: {column_names}

    Please generate 3 interesting and distinct questions that a data analyst might ask.
    The questions should be suitable to be answered using Python's pandas library.
    Return the questions as a Python list of strings. For example: ["What is the total sales?", "Who are the top 5 customers?", "What is the monthly sales trend?"]
    """
    try:
        response = model.generate_content(prompt)
        # Use ast.literal_eval for safety to convert string list to Python list
        questions = ast.literal_eval(response.text)
        return questions
    except (ValueError, SyntaxError, AttributeError):
        # Fallback questions if LLM generation fails or returns malformed string
        return [
            f"What is the distribution of values in the '{df.columns[0]}' column?",
            f"Provide a statistical summary of the numerical columns.",
            f"How many unique values are in each categorical column?"
        ]

# --- Streamlit UI ---

# Set page configuration
st.set_page_config(
    page_title="📈 AI Agent Data Analyst",
    page_icon="🤖",
    layout="centered"
)

st.title("📈 Data Analyst AI Agent")

# Initialize session state for all necessary variables
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Upload a CSV or Excel file to get started."}]
if "agent" not in st.session_state:
    st.session_state.agent = None
if "dataframe" not in st.session_state:
    st.session_state.dataframe = None
if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = []
if "prompt_from_button" not in st.session_state:
    st.session_state.prompt_from_button = None

# --- Sidebar for File Upload and Data Cleaning ---
with st.sidebar:
    st.header("1. Setup")
    uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        # Load the dataframe once
        if st.session_state.dataframe is None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    st.session_state.dataframe = pd.read_csv(uploaded_file)
                else:
                    st.session_state.dataframe = pd.read_excel(uploaded_file)
                # Clear previous agent and questions if a new file is uploaded
                st.session_state.agent = None
                st.session_state.suggested_questions = []
            except Exception as e:
                st.error(f"Error loading file: {e}")
                st.session_state.dataframe = None

    if st.session_state.dataframe is not None:
        st.subheader("2. Data Cleaning (Optional)")
        df = st.session_state.dataframe
        
        # Display a preview
        st.write("**Data Preview:**")
        st.dataframe(df.head())

        # Data cleaning options
        st.markdown("**Handle Missing Values**")
        missing_strategy = st.selectbox("Strategy for numerical columns", ["None", "Fill with Mean", "Fill with Median", "Drop Rows with any NaNs"])

        if st.button("Apply Cleaning"):
            with st.spinner("Applying cleaning..."):
                cleaned_df = df.copy()
                if missing_strategy != "None":
                    num_cols = cleaned_df.select_dtypes(include="number").columns
                    if missing_strategy == "Fill with Mean":
                        cleaned_df[num_cols] = cleaned_df[num_cols].fillna(cleaned_df[num_cols].mean())
                    elif missing_strategy == "Fill with Median":
                        cleaned_df[num_cols] = cleaned_df[num_cols].fillna(cleaned_df[num_cols].median())
                    elif missing_strategy == "Drop Rows with any NaNs":
                        cleaned_df.dropna(inplace=True)
                
                st.session_state.dataframe = cleaned_df
                st.success("Cleaning applied!")
                # Force a rerun to show the updated dataframe preview
                st.rerun()

        st.subheader("3. Create Agent")
        if st.button("Create Analysis Agent"):
            with st.spinner("Creating agent and generating suggestions..."):
                st.session_state.agent = create_agent(st.session_state.dataframe)
                st.session_state.suggested_questions = generate_suggested_questions(st.session_state.dataframe)
                st.success("Agent is ready!", icon="✅")
                # Rerun to update the main page UI
                st.rerun()

    # Add a reset button
    st.header("Reset")
    if st.button("Reset Chat & Data"):
        # Clear all session state variables to start fresh
        for key in st.session_state.keys():
            del st.session_state[key]
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

# --- Main Chat Interface ---

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "plot" in message:
            st.image(message["plot"])
            with open(message["plot"], "rb") as f:
                st.download_button(
                    label="Download Chart",
                    data=f,
                    file_name="plot.png",
                    mime="image/png"
                )

# Display suggested questions if the agent is ready
if st.session_state.agent is not None and st.session_state.suggested_questions and len(st.session_state.messages) == 1:
    st.markdown("---")
    st.subheader("💡 Here are some ideas to get you started:")
    cols = st.columns(len(st.session_state.suggested_questions))
    for i, question in enumerate(st.session_state.suggested_questions):
        with cols[i]:
            if st.button(question, key=f"suggestion_{i}"):
                st.session_state.prompt_from_button = question
                st.rerun()

# Handle input from either the chat input box or a clicked suggestion button
prompt = st.chat_input("Ask a question about your data...")
if st.session_state.prompt_from_button:
    prompt = st.session_state.prompt_from_button
    st.session_state.prompt_from_button = None  # Reset after use

if prompt:
    if st.session_state.agent is None:
        st.warning("Please upload a file and create an agent first in the sidebar.", icon="⚠️")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if os.path.exists("plot.png"):
                    os.remove("plot.png")

                try:
                    chat_history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
                    query_with_context = f"Conversation history:\n{chat_history_str}\n\nUser's new question: {prompt}"

                    # Capture terminal output
                    stdout_capture = io.StringIO()
                    with redirect_stdout(stdout_capture):
                        response = st.session_state.agent.query(query_with_context)
                    captured_output = stdout_capture.getvalue()

                    raw_answer = str(response).strip()
                    generated_code = response.metadata.get('pandas_instruction_str', '')

                    final_answer = ""
                    assistant_response = {"role": "assistant"}
                    
                    # Case 1: A plot was generated successfully.
                    if os.path.exists("plot.png"):
                        final_answer = "I've generated the requested chart for you."
                        st.markdown(final_answer)
                        st.image("plot.png")
                        assistant_response["plot"] = "plot.png"
                    
                    # Case 2: The agent intended a text answer (code execution failed with a syntax error).
                    elif ("invalid syntax" in captured_output or "error running the output" in captured_output) and generated_code:
                        final_answer = generated_code
                        st.markdown(final_answer)

                    # Case 3: Code was executed and produced a text/data output.
                    else:
                        # Parse the captured output to remove verbose logging
                        result_output = ""
                        if "> Pandas Instructions:" in captured_output:
                            parts = captured_output.split("> Pandas Output:")
                            printed_result_block = parts[0]
                            cleaned_block = printed_result_block.replace("> Pandas Instructions:", "").strip()
                            printed_result = cleaned_block.replace(generated_code.strip(), "").strip()
                            returned_result = ""
                            if len(parts) > 1:
                                returned_result = parts[1].strip()
                            if printed_result:
                                result_output = printed_result
                            elif returned_result and returned_result != "None":
                                result_output = returned_result
                        else:
                            result_output = captured_output.strip()

                        # Assemble the final answer from the parts
                        final_answer_parts = []
                        if raw_answer and raw_answer != "None":
                            final_answer_parts.append(raw_answer)
                        if result_output:
                            final_answer_parts.append(f"```\n{result_output}\n```")
                        
                        if not final_answer_parts:
                            final_answer = "I've completed the action. There's no specific text output to show."
                        else:
                            final_answer = "\n\n".join(final_answer_parts)
                        
                        st.markdown(final_answer)

                    # Final cleanup for any strange artifacts
                    final_answer = final_answer.replace("undefined", "").replace("undefined", "").replace("undefined", "").strip()
                    
                    # Display the generated code in the expander
                    with st.expander("🔎 Show Generated Code"):
                        st.code(generated_code, language="python")

                    # Save the final, clean answer to the chat history
                    assistant_response["content"] = final_answer
                    st.session_state.messages.append(assistant_response)

                except Exception as e:
                    error_message = f"Sorry, I encountered an error: {e}"
                    st.error(error_message, icon="🚨")
                    st.session_state.messages.append({"role": "assistant", "content": error_message})