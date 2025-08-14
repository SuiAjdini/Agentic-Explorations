import os
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, Document, PromptTemplate
from llama_index.llms.gemini import Gemini
from cv_parser import config
from cv_parser.models import CandidateProfile, FeedbackIntent

class CVExtractor:
    """
    Manages loading CVs, extracting information, and handling feedback.
    """
    def __init__(self):
        # Set API key from config
        os.environ["GOOGLE_API_KEY"] = config.GOOGLE_API_KEY

        # --- 1. Initial Settings ---
        self.llm = Gemini(model_name=config.LLM_MODEL_NAME)
        Settings.llm = self.llm
        Settings.embed_model = config.EMBED_MODEL_NAME
        
        # --- 2. Load Documents and Set Up Index ---
        self._setup_index()

    def _setup_index(self):
        """Loads data and initializes the vector store index and query engine."""
        try:
            reader = SimpleDirectoryReader(config.CV_DATA_PATH)
            # Combine all documents into a single Document object for better context
            initial_documents = [Document(text="\n\n".join([doc.text for doc in reader.load_data()]))]
            self.index = VectorStoreIndex.from_documents(initial_documents)
            self.query_engine = self.index.as_query_engine(output_cls=CandidateProfile, verbose=False)
        except Exception as e:
            print(f"❌ Error setting up the index: {e}")
            raise

    def extract_initial_profile(self) -> CandidateProfile:
        """Runs the initial extraction query."""
        prompt_query = "Extract the key information from the provided CV document. Please fill in all the fields of the CandidateProfile schema as accurately as possible based on the document's content."
        print("\n🚀 Querying the agent for initial extraction...")
        response = self.query_engine.query(prompt_query)
        return response.response

    def update_index_with_correction(self, corrected_json_str: str) -> CandidateProfile:
        """Updates the index with user-corrected data and re-runs the query."""
        print("\n🧠 Thank you! Learning from your feedback by updating the index...")
        corrected_doc = Document(
            text=f"This is a human-verified, correct extraction: {corrected_json_str}",
            metadata={"source": "human_feedback"}
        )
        self.index.insert_documents([corrected_doc])
        
        print("Index updated. Re-running extraction to apply correction...")
        return self.extract_initial_profile()

    def analyze_feedback(self, user_input: str) -> FeedbackIntent:
        """Uses the LLM to classify the user's feedback into a specific intent."""
        prompt = PromptTemplate("""
            A user has provided the following feedback on extracted data: '{user_input}'.
            Analyze this feedback and classify it into one of the following intents:
            - 'confirm': The user is happy with the data.
            - 'correct': The user wants to correct a mistake in the data.
            - 'query': The user is asking a question about the data.
            - 'unclear': The user's intent cannot be determined.
            Provide your analysis in the requested format.
            """)
        try:
            return self.llm.structured_predict(FeedbackIntent, prompt, user_input=user_input)
        except Exception as e:
            print(f"Error analyzing feedback: {e}")
            return FeedbackIntent(intent="unclear", details=None)

    def answer_query_on_data(self, data_json: str, user_query: str) -> str:
        """Uses the LLM to answer a user's question based on the provided JSON data."""
        prompt = PromptTemplate("""
            Given the following JSON data:
            ---
            {data_json}
            ---
            Answer the user's question based *only* on this data.
            User question: '{user_query}'
            """)
        return self.llm.predict(prompt, data_json=data_json, user_query=user_query)