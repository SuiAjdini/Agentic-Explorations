import gradio as gr
import json
import google.generativeai as genai
from cv_parser.extractor import CVExtractor
from cv_parser import config

# --- 1. Configure the Gemini API Client ---

try:
    print("🔑 Configuring Gemini API...")
    genai.configure(api_key=config.GOOGLE_API_KEY)
    print("✅ Gemini API configured.")
except Exception as e:
    print(f"❌ Error configuring Gemini API: {e}")

# --- 2. Define Helper and Core Gradio Functions ---

def transcribe_audio_with_gemini(audio_path: str) -> str:
    """Transcribes the given audio file using the Gemini 1.5 Flash model."""
    if not audio_path:
        return ""
    try:
        print(f"⬆️ Uploading audio file to Gemini: {audio_path}")
        audio_file = genai.upload_file(path=audio_path)

        print("🤖 Requesting transcription from Gemini...")
        model = genai.GenerativeModel('models/gemini-1.5-flash-latest')
        response = model.generate_content(
            ["Please transcribe this audio recording accurately.", audio_file],
            request_options={"timeout": 120} # Set a 2-minute timeout
        )

        # Clean up the uploaded file after processing
        genai.delete_file(audio_file.name)
        print("✅ Transcription received.")
        return response.text.strip()
    except Exception as e:
        print(f"❌ Error during transcription: {e}")
        return f"Error: Could not transcribe audio. {e}"


def load_and_extract():
    """
    This function runs once when the Gradio app loads.
    It initializes the CVExtractor and performs the first extraction.
    """
    print("🚀 Initializing CV Extractor and performing initial extraction...")
    try:
        extractor = CVExtractor()
        initial_profile = extractor.extract_initial_profile()
        initial_json = json.loads(initial_profile.model_dump_json())
        print("✅ Initial extraction complete.")
        return extractor, initial_json, "Extraction complete. Please provide feedback."
    except Exception as e:
        print(f"❌ Failed to initialize or extract: {e}")
        error_message = f"Error during startup: {e}"
        return None, {"error": error_message}, error_message

def process_voice_feedback(audio_input, extractor, current_profile_json):
    """
    Handles the voice input from the user by transcribing it with Gemini.
    """
    if extractor is None:
        return "Extractor not loaded. Please restart.", current_profile_json

    if audio_input is None:
        return "Please record your feedback first.", current_profile_json

    # --- Step 1: Transcribe Audio to Text with Gemini ---
    user_feedback_text = transcribe_audio_with_gemini(audio_input)
    if user_feedback_text.startswith("Error:"):
        return user_feedback_text, current_profile_json # Display the error
    print(f"💬 User said: '{user_feedback_text}'")

    # --- Step 2: Analyze Intent ---
    print("🧠 Analyzing intent...")
    intent_analysis = extractor.analyze_feedback(user_feedback_text)
    intent = intent_analysis.intent
    details = intent_analysis.details

    # --- Step 3: Act on Intent ---
    current_profile_str = json.dumps(current_profile_json, indent=4)

    if intent == "confirm":
        return "✅ Great! Process confirmed. Data is ready.", current_profile_json
    
    elif intent == "query":
        print(f"❓ Answering query: '{details}'")
        answer = extractor.answer_query_on_data(current_profile_str, details)
        return f"❓ Query: {details}\n\nAnswer: {answer}", current_profile_json

    elif intent == "correct":
        return "✍️ Correction needed. Please edit the JSON in the box below and click 'Submit Manual Correction'.", current_profile_json
    
    else: # unclear
        return "❓ I'm not sure what you mean. Could you please rephrase?", current_profile_json

def process_manual_correction(edited_json, extractor):
    """
    Handles the submission of a manually corrected JSON.
    """
    if extractor is None:
        return "Extractor not loaded. Please restart.", edited_json
    
    print("🔄 Processing manual correction...")
    try:
        corrected_json_str = json.dumps(edited_json, indent=4)
        updated_profile = extractor.update_index_with_correction(corrected_json_str)
        new_json_output = json.loads(updated_profile.model_dump_json())
        print("✅ Correction processed and index updated.")
        return "✅ Correction applied and index updated. Re-running extraction...", new_json_output
    except Exception as e:
        print(f"❌ Error processing correction: {e}")
        return f"Error applying correction: {e}", edited_json


# --- 3. Build the Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft(), title="CV Parsing Agent") as app:
    # ... (The UI layout is exactly the same as before)
    gr.Markdown("# 🤖 CV Parsing Agent with Voice Feedback")
    gr.Markdown("The agent automatically extracts information from a CV. Review the extracted JSON below and provide feedback using your voice.")

    extractor_state = gr.State()
    json_state = gr.State()

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Extracted CV Data")
            json_output = gr.JSON(label="Extracted Information")
            
        with gr.Column(scale=1):
            gr.Markdown("## Your Feedback")
            audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Record Your Feedback")
            submit_voice_btn = gr.Button("Process Voice Feedback", variant="primary")
            submit_correction_btn = gr.Button("Submit Manual Correction")
            
            gr.Markdown("---")
            status_box = gr.Textbox(label="Status & Agent Responses", lines=5)

    # --- 4. Connect the components ---
    app.load(fn=load_and_extract, inputs=[], outputs=[extractor_state, json_output, status_box])
    submit_voice_btn.click(fn=process_voice_feedback, inputs=[audio_input, extractor_state, json_output], outputs=[status_box, json_output])
    submit_correction_btn.click(fn=process_manual_correction, inputs=[json_output, extractor_state], outputs=[status_box, json_output])


# --- 5. Launch the App ---
if __name__ == "__main__":
    app.launch()