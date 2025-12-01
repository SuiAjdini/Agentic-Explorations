from firecrawl import Firecrawl 
import streamlit as st
import os
import json
from agno.agent import Agent
from agno.models.google import Gemini
from dotenv import load_dotenv

# 1. Load environment variables
load_dotenv()

st.set_page_config(
    page_title="Startup Info Extraction",
    page_icon="🔍",
    layout="wide"
)

st.title("AI Startup Insight with Firecrawl & Gemini")

# Sidebar for API key
with st.sidebar:
    st.header("API Configuration")
    
    # 2. Logic to handle .env vs Manual Input
    
    # Check Firecrawl Key
    env_firecrawl = os.getenv("FIRECRAWL_API_KEY")
    if env_firecrawl:
        st.success("✅ Firecrawl Key loaded from .env")
        firecrawl_api_key = env_firecrawl
    else:
        firecrawl_api_key = st.text_input("Firecrawl API Key", type="password")

    # Check Google Key
    env_google = os.getenv("GOOGLE_API_KEY")
    if env_google:
        st.success("✅ Google Key loaded from .env")
        google_api_key = env_google
    else:
        google_api_key = st.text_input("Google Gemini API Key", type="password")
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This tool extracts company information using Firecrawl and analyzes it using **Google Gemini**.")
    

# Main content
st.markdown("## 🔥 Firecrawl FIRE 1 Agent Capabilities")

col1, col2 = st.columns(2)

with col1:
    st.info("**Advanced Web Extraction**\n\nFirecrawl's FIRE 1 agent navigates websites to extract structured data, even from complex layouts.")

with col2:
    st.warning("**Multi-page Processing**\n\nFIRE handles pagination to gather comprehensive data across entire websites.")

st.markdown("---")

st.markdown("### 🌐 Enter Website URLs")
website_urls = st.text_area("Website URLs (one per line)", placeholder="https://example.com\nhttps://another-company.com")

# Define JSON schema
extraction_schema = {
    "type": "object",
    "properties": {
        "company_name": { "type": "string", "description": "The official name of the company" },
        "company_description": { "type": "string", "description": "What the company does and its value proposition" },
        "company_mission": { "type": "string", "description": "The company's mission statement" },
        "product_features": { 
            "type": "array", 
            "items": { "type": "string" }, 
            "description": "Key features of products/services" 
        },
        "contact_phone": { "type": "string", "description": "Contact phone number if available" }
    },
    "required": ["company_name", "company_description", "product_features"]
}

# Custom CSS
st.markdown("""
<style>
.stButton button {
    background-color: #4285F4;
    color: white;
    font-weight: bold;
    border-radius: 10px;
    padding: 0.5rem 1rem;
    border: none;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease;
}
.stButton button:hover {
    background-color: #2b6cb0;
    box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    transform: translateY(-2px);
}
</style>
""", unsafe_allow_html=True)

# Start extraction
if st.button("🚀 Start Analysis", type="primary"):
    if not website_urls.strip():
        st.error("Please enter at least one website URL")
    else:
        try:
            with st.spinner("Preparing agents..."):
                # Initialize Firecrawl with the loaded key
                app = Firecrawl(api_key=firecrawl_api_key)
                urls = [url.strip() for url in website_urls.split('\n') if url.strip()]
                
                if not urls:
                    st.error("No valid URLs found.")
                elif not google_api_key:
                    st.warning("Please provide a Google Gemini API key.")
                else:
                    tabs = st.tabs([f"Website {i+1}: {url}" for i, url in enumerate(urls)])
                    
                    # Initialize Gemini Agent with the loaded key
                    if google_api_key:
                        agno_agent = Agent(
                            model=Gemini(id="gemini-2.5-flash", api_key=google_api_key),
                            instructions="""You are an expert business analyst.
                            You will be given structured data about a company.
                            Analyze this information and provide a brief summary highlighting:
                            1. Unique innovation
                            2. Core value proposition
                            3. Potential market impact
                            
                            Keep response under 150 words. Be specific.
                            """,
                            markdown=True
                        )
                    
                    for i, (url, tab) in enumerate(zip(urls, tabs)):
                        with tab:
                            st.markdown(f"### 🔍 Analyzing: {url}")
                            st.markdown("<hr style='border: 2px solid #4285F4; border-radius: 5px;'>", unsafe_allow_html=True)
                            
                            with st.spinner(f"FIRE agent extracting from {url}..."):
                                try:
                                    data = app.extract(
                                        urls=[url],
                                        prompt="Analyze this website and extract company info, mission, and features.",
                                        schema=extraction_schema,
                                        agent={"model": "FIRE-1"},
                                    )
    
                                    
                                    if data and getattr(data, "data", None):
                                        st.subheader("📊 Extracted Information")
                                        company_data = data.data  # instead of data.get('data')

                                        if 'company_name' in company_data:
                                            st.markdown(f"### {company_data['company_name']}")
                                                                            
                                        for key, value in company_data.items():
                                            if key == 'company_name': continue
                                            display_key = key.replace('_', ' ').capitalize()
                                            if value:
                                                if isinstance(value, list):
                                                    st.markdown(f"**{display_key}:**")
                                                    for item in value: st.markdown(f"- {item}")
                                                else:
                                                    st.markdown(f"**{display_key}:** {value}")
                                        
                                        if google_api_key:
                                            with st.spinner("Generating Gemini analysis..."):
                                                agent_response = agno_agent.run(
                                                            f"Analyze this company data: {json.dumps(company_data)}"
                                                        )
                                                                                                        
                                                st.subheader("🧠 Gemini Business Analysis")
                                                st.markdown(agent_response.content)
                                        
                                        with st.expander("🔍 View Raw Data"):
                                            st.json(data)
                                    else:
                                        st.error(f"No data extracted from {url}.")
                                        
                                except Exception as e:
                                    st.error(f"Error processing {url}: {str(e)}")
        except Exception as e:
            st.error(f"Error during extraction: {str(e)}")