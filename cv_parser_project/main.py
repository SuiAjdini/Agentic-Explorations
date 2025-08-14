import json
from cv_parser.extractor import CVExtractor
from cv_parser import config

def get_multiline_json_input():
    """Gets multiline input from the user until they press Enter on an empty line."""
    print("Copy the JSON above, fix it, and paste it here. Press Enter on an empty line to finish.")
    lines = []
    while True:
        line = input()
        if not line:
            break
        lines.append(line)
    return "\n".join(lines)

def main():
    """Main conversational loop for CV extraction and correction."""
    try:
        extractor = CVExtractor()
    except Exception as e:
        print(f"Failed to initialize the CV Extractor. Exiting. Error: {e}")
        return

    # Initial extraction
    extracted_profile = extractor.extract_initial_profile()
    
    while True:
        extracted_json = extracted_profile.model_dump_json(indent=4)
        print("\n--- Extracted Data ---")
        print(extracted_json)

        user_feedback = input("\n🤔 What do you think? If it's correct, say so. Otherwise, tell me what to change or ask a question: ").strip()

        if not user_feedback:
            continue

        intent_analysis = extractor.analyze_feedback(user_feedback)
        intent = intent_analysis.intent

        if intent == "confirm":
            print("\n✅ Great! Process complete.")
            with open(config.FINAL_OUTPUT_FILENAME, "w") as f:
                f.write(extracted_json)
            print(f"Final data saved to '{config.FINAL_OUTPUT_FILENAME}'")
            break

        elif intent == "query":
            print(f"\n💬 Answering your query: '{intent_analysis.details}'")
            answer = extractor.answer_query_on_data(extracted_json, intent_analysis.details)
            print(f"\nAnswer: {answer}")
            input("\nPress Enter to continue...")

        elif intent == "correct":
            corrected_json_str = get_multiline_json_input()
            try:
                # Validate that the user provided valid JSON
                json.loads(corrected_json_str) 
                extracted_profile = extractor.update_index_with_correction(corrected_json_str)
            except json.JSONDecodeError:
                print("\n❌ Error: The text you pasted is not valid JSON. Please try again.")

        else: # Unclear intent
            print("\n❓ I'm not sure what you mean. Could you please rephrase?")

if __name__ == "__main__":
    main()