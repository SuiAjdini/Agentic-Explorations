import streamlit as st
import os
from dotenv import load_dotenv

# Core LangChain components
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad.tools import format_to_tool_messages
from langchain.agents.output_parsers.tools import ToolsAgentOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

def create_shopping_agent():
    """
    Initializes and returns the shopping agent by manually constructing the agent runnable.
    This is a stable method that relies on core LangChain components.
    """
    
    # 1. Set up the LLM and Tools
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.7)
    search_tool = TavilySearchResults(max_results=3)
    tools = [search_tool]
    
    # 2. Bind the tools to the LLM
    # This makes the LLM aware of the functions it can call.
    llm_with_tools = llm.bind_tools(tools)
    
    # 3. Create the Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful and expert product researcher."),
        ("user", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    # 4. Create the Agent Runnable using LangChain Expression Language (LCEL)
    # This defines the agent's logic flow.
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_tool_messages(x["intermediate_steps"]),
        }
        | prompt
        | llm_with_tools
        | ToolsAgentOutputParser()
    )
    
    # 5. Create the Agent Executor to run the agent
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True
    )
    
    return agent_executor

# --- STREAMLIT UI ---
st.set_page_config(page_title="Smart Shopper Agent", page_icon="🛍️")

st.title("🛍️ Smart Shopper Agent")
st.markdown("Your personal AI assistant to find the best products for you!")

query = st.text_input("What are you looking for today?", placeholder="e.g., best noise-canceling headphones under $200")

if query:
    with st.spinner("🤖 Searching the web and analyzing reviews for you..."):
        try:
            shopping_agent = create_shopping_agent()
            
            prompt_for_agent = f"""
            Find the best product for the user based on their query: '{query}'.

            Follow these steps:
            1) Use the web-search tool to find recent reviews, comparisons, pricing, and **product listings from reputable shops**.
            2) Identify 2–3 top contenders that fit the user's intent.
            3) For each contender:
            - Summarize key features and notable pros/cons (concise).
            - Provide **2–3 purchase links** from reputable shops (prefer EU/DE like Amazon.de, MediaMarkt, Saturn, Idealo, Thomann, etc.). 
            - If possible, include **current price** and **availability** from the page you cite.
            - Use **clean, canonical URLs** (no tracking params if you can avoid them).

            Formatting (Markdown):
            - **🏆 Top Picks:** A short bullet list (product name + one-line why)
            - **✨ Product Details:** For each product, use this template:

            ### {{Product Name}}
            **Why it’s good:** one-liner  
            **Key specs:** short bullets  
            **Pros:** short bullets  
            **Cons:** short bullets  
            **Where to buy:**
            - [Shop Name 1](https://...) — Price (if visible)
            - [Shop Name 2](https://...) — Price (if visible)
            - [Shop Name 3](https://...) — Price (optional)

            - **🤔 Final Verdict:** Which one you recommend and why.

            Guidelines:
            - Prefer **official product pages or major retailers**. Avoid forums or random blogs for shop links.
            - If multiple model variants exist, make sure the shop link matches the **exact model** you’re recommending.
            - If price is missing, write “Price not shown”.
            - Be transparent: “Prices and availability can change.”
            """


            response = shopping_agent.invoke({"input": prompt_for_agent})
            st.markdown(response['output'])

        except Exception as e:
            st.error(f"An error occurred: {e}")