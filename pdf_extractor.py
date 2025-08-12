import os
import json
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, Document, PromptTemplate
from llama_index.llms.gemini import Gemini
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

os.environ["GOOGLE_API_KEY"] = "API_KEY" 


# --- 1. Initial Settings ---
# This tells llama_index to use Gemini as the LLM
llm = Gemini(model_name="models/gemini-1.5-flash-latest")
Settings.llm = llm

# This sets up a local model for creating embeddings
Settings.embed_model = "local:BAAI/bge-small-en-v1.5"

# --- 2. Define the Desired JSON Structure using Pydantic ---
# We are creating a "schema" that tells the AI exactly what fields to extract.

class WorkExperience(BaseModel):
    """Data model for a single work experience."""
    job_title: str = Field(description="The job title or position.")
    company: str = Field(description="The name of the company.")
    location: Optional[str] = Field(description="The location of the company (e.g., city, state).")
    start_date: Optional[str] = Field(description="The start date of the employment.")
    end_date: Optional[str] = Field(description="The end date of the employment. Can be 'Present'.")
    description: Optional[str] = Field(description="A brief description of the role and responsibilities.")

class Education(BaseModel):
    """Data model for a single education entry."""
    institution: str = Field(description="The name of the educational institution.")
    degree: str = Field(description="The degree obtained (e.g., Bachelor of Science).")
    field_of_study: Optional[str] = Field(description="The field of study (e.g., Computer Science).")
    graduation_date: Optional[str] = Field(description="The date of graduation.")

class CandidateProfile(BaseModel):
    """The main data model for the extracted CV information."""
    name: str = Field(description="The full name of the candidate.")
    email: Optional[str] = Field(description="The email address of the candidate.")
    phone: Optional[str] = Field(description="The phone number of the candidate.")
    summary: Optional[str] = Field(description="A brief summary or objective from the CV.")
    skills: List[str] = Field(description="A list of skills mentioned in the CV.")
    work_experience: List[WorkExperience] = Field(description="A list of the candidate's work experiences.")
    education: List[Education] = Field(description="A list of the candidate's educational background.")

class FeedbackIntent(BaseModel):
    """Data model for classifying the user's feedback intent."""
    intent: Literal["confirm", "correct", "query", "unclear"] = Field(
        description="The user's intent. Must be 'confirm' for approval, 'correct' for fixing errors, 'query' for asking questions about the data, or 'unclear'."
    )
    details: Optional[str] = Field(description="If intent is 'query' or 'correct', this contains the user's specific request or the identified error.")

# --- 3. Helper Functions for Conversational Flow ---

def analyze_feedback(user_input: str, llm_instance: Gemini) -> FeedbackIntent:
    """Uses the LLM to classify the user's feedback into a specific intent."""
    prompt_template = PromptTemplate("""
        A user has provided the following feedback on extracted data: '{user_input}'.
        Analyze this feedback and classify it into one of the following intents:
        - 'confirm': The user is happy with the data.
        - 'correct': The user wants to correct a mistake in the data.
        - 'query': The user is asking a question about the data.
        - 'unclear': The user's intent cannot be determined.

        Provide your analysis in the requested format.
        """)

    try:
        response = llm_instance.structured_predict(
            FeedbackIntent,
            prompt_template,
            user_input=user_input
        )
        return response
    except Exception as e:
        print(f"Error analyzing feedback: {e}")
        return FeedbackIntent(intent="unclear", details=None)


def answer_query_on_data(data_json: str, user_query: str, llm_instance: Gemini) -> str:
    """Uses the LLM to answer a user's question based on the provided JSON data."""
    
    # Define a formal template with placeholders
    prompt_template = PromptTemplate("""
Given the following JSON data:
---
{data_json}
---
Answer the user's question based *only* on this data.
User question: '{user_query}'
""")
    
    # Use the template with the .predict() method for a direct string answer
    response = llm_instance.predict(
        prompt_template,
        data_json=data_json,
        user_query=user_query
    )
    return response

# --- 4. Load Documents and Set Up Index ---
try:
    reader = SimpleDirectoryReader("./cv_data")
    initial_documents = [Document(text="\n\n".join([doc.text for doc in reader.load_data()]))]
    index = VectorStoreIndex.from_documents(initial_documents)
    query_engine = index.as_query_engine(output_cls=CandidateProfile, verbose=False)
except Exception as e:
    print(f"Error loading documents: {e}")
    exit()

# --- 5. Main Conversational Loop ---
prompt_query = "Extract the key information from the provided CV document. Please fill in all the fields of the CandidateProfile schema as accurately as possible based on the document's content."

print("\n🚀 Querying the agent for initial extraction...")
response = query_engine.query(prompt_query)
extracted_object = response.response
extracted_json = extracted_object.model_dump_json(indent=4)

while True:
    print("\n--- Extracted Data ---")
    print(extracted_json)

    user_feedback = input("\n🤔 What do you think? If it's correct, say so. Otherwise, tell me what to change or ask a question: ").strip()

    if not user_feedback:
        continue

    # Analyze the user's intent
    intent_analysis = analyze_feedback(user_feedback, llm)
    intent = intent_analysis.intent

    if intent == "confirm":
        print("\n✅ Great! Process complete.")
        output_filename = "extracted_info_final.json"
        with open(output_filename, "w") as f:
            f.write(extracted_json)
        print(f"Final data saved to '{output_filename}'")
        break

    elif intent == "query":
        print(f"\n💬 Answering your query: '{intent_analysis.details}'")
        answer = answer_query_on_data(extracted_json, intent_analysis.details, llm)
        print(f"\nAnswer: {answer}")
        input("\nPress Enter to continue...") # Pause to allow user to read

    elif intent == "correct":
        print("\n✍️ It looks like there's a mistake. Please provide the corrected information.")
        print("Copy the JSON above, fix it, and paste the entire corrected JSON here, then press Enter twice.")
        
        corrected_lines = []
        while True:
            line = input()
            if not line: break
            corrected_lines.append(line)
        corrected_json_str = "\n".join(corrected_lines)

        try:
            json.loads(corrected_json_str)
            print("\n🧠 Thank you! Learning from your feedback by updating the index...")
            corrected_doc = Document(text=f"This is a human-verified, correct extraction: {corrected_json_str}", metadata={"source": "human_feedback"})
            index.insert_documents([corrected_doc])
            
            print("Index updated. Re-running extraction to apply correction...")
            response = query_engine.query(prompt_query) # Re-query
            extracted_object = response.response
            extracted_json = extracted_object.model_dump_json(indent=4)

        except json.JSONDecodeError:
            print("\n❌ Error: The text you pasted is not valid JSON. Please try again.")

    else: # unclear
        print("\n❓ I'm not sure what you mean. Could you please rephrase your feedback?")
