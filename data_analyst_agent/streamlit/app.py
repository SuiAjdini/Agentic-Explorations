import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import streamlit as st
import google.generativeai as genai

from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.llms.gemini import Gemini
import config

# --- Gemini API Configuration ---
try:
    genai.configure(api_key=config.GOOGLE_API_KEY)
except Exception as e:
    st.error(f"❌ Error configuring Gemini API: {e}", icon="🚨")

# --- Core Functions ---
@st.cache_resource
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

# --- Streamlit UI ---

# Set page configuration
st.set_page_config(
    page_title="📈 AI Agent Data Analyst",
    page_icon="🤖",
    layout="centered"
)

st.title("📈 Data Analyst AI Agent")

# Initialize session state for chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Upload a CSV file to get started and I'll help you analyze it."}]

# Initialize session state for the agent and dataframe
if "agent" not in st.session_state:
    st.session_state.agent = None
if "dataframe" not in st.session_state:
    st.session_state.dataframe = None

# --- Sidebar for File Upload ---
with st.sidebar:
    st.header("Setup")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None and st.session_state.agent is None:
        with st.spinner("Processing CSV and creating agent..."):
            try:
                # Load the dataframe
                df = pd.read_csv(uploaded_file)
                st.session_state.dataframe = df
                
                # Create and store the agent
                st.session_state.agent = create_agent(df)
                
                st.success("Agent is ready! You can now ask questions about your data.", icon="✅")
                
                # Show a preview of the data
                st.subheader("Data Preview:")
                st.dataframe(df.head())
                
            except Exception as e:
                st.error(f"Error processing file: {e}", icon="🚨")
                st.session_state.agent = None # Reset agent on error
                st.session_state.dataframe = None

    # Add a reset button
    if st.button("Reset Chat & Data"):
        st.session_state.messages = [{"role": "assistant", "content": "Hello! Upload a CSV file to get started and I'll help you analyze it."}]
        st.session_state.agent = None
        st.session_state.dataframe = None
        # Clear the cache to allow for new agent creation
        st.cache_resource.clear()
        st.rerun()

# --- Main Chat Interface ---

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If a plot exists for an assistant message, display it
        if "plot" in message:
            st.image(message["plot"])
            with open(message["plot"], "rb") as f:
                st.download_button(
                    label="Download Chart",
                    data=f,
                    file_name="plot.png",
                    mime="image/png"
                )


# Chat input box at the bottom of the page
if prompt := st.chat_input("Ask a question about your data..."):
    # Check if agent is ready
    if st.session_state.agent is None:
        st.warning("Please upload and load a CSV file first.", icon="⚠️")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Clean up previous plot if it exists
                if os.path.exists("plot.png"):
                    os.remove("plot.png")

                try:
                    # Prepare the query with conversation history
                    chat_history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
                    query_with_context = f"Conversation history:\n{chat_history_str}\n\nUser's new question: {prompt}"

                    # Query the agent
                    response = st.session_state.agent.query(query_with_context)
                    raw_answer = str(response)

                    final_answer = ""
                    assistant_response = {"role": "assistant"}

                    # Check if a plot was created
                    if os.path.exists("plot.png"):
                        final_answer = "I've generated the requested chart for you."
                        st.markdown(final_answer)
                        st.image("plot.png")
                        assistant_response["content"] = final_answer
                        assistant_response["plot"] = "plot.png" # Store path to show download button
                        with open("plot.png", "rb") as f:
                            st.download_button(
                                label="Download Chart",
                                data=f,
                                file_name="plot.png",
                                mime="image/png"
                            )

                    # If no plot, process text response
                    else:
                        if raw_answer == "None":
                            final_answer = "I've completed the action, but there's no specific data to show."
                        elif 'count' in raw_answer and 'mean' in raw_answer:
                            try:
                                summarizer_model = genai.GenerativeModel(config.LLM_MODEL_NAME)
                                summary_prompt = f"Please summarize the following statistical data in a friendly, natural language paragraph:\n\n{raw_answer}"
                                summary_response = summarizer_model.generate_content(summary_prompt)
                                final_answer = summary_response.text
                            except Exception:
                                final_answer = "I found some statistics about your data, but couldn't summarize them:\n\n" + raw_answer
                        else:
                            final_answer = raw_answer
                        
                        st.markdown(final_answer)
                        assistant_response["content"] = final_answer
                    
                    st.session_state.messages.append(assistant_response)

                except Exception as e:
                    error_message = f"Sorry, I encountered an error: {e}"
                    st.error(error_message, icon="🚨")
                    st.session_state.messages.append({"role": "assistant", "content": error_message})