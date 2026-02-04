import requests
import json
import argparse
import os
from datetime import datetime

BASE_URL = "http://localhost:8080/chatbot/v1/chat"

def create_headers(email_id):
    return {
        'x-digihub-emailid': email_id,
        'x-digihub-tenantid': '1',
        'x-digihub-traceid': '123456789',
        'authorization': 'Bearer '
    }


def load_questions(filepath):
    with open(filepath, 'r') as f:
        questions = [line.strip() for line in f if line.strip()]
    return questions


def invoke_question(query, session_id="", headers=None):
    payload = {
        "chat_session_id": session_id,
        "query": query
    }
    response = requests.post(BASE_URL, json=payload, headers=headers)
    return response.json()


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Invoke chatbot questions from a file')
    parser.add_argument('--input-file', '-i', default='questions.md',
                        help='Input file containing questions (default: questions.md)')
    parser.add_argument('--email', '-e', default='sameer.kumar@sita.aero',
                        help='Email ID for x-digihub-emailid header (default: sameer.kumar@sita.aero)')
    args = parser.parse_args()
    
    # Create headers with provided email
    headers = create_headers(args.email)
    
    # Create output file name: input_file_name_email_timestamp.json
    input_basename = os.path.splitext(os.path.basename(args.input_file))[0]
    email_prefix = args.email.split('@')[0].replace('.', '_')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"{input_basename}_{email_prefix}_{timestamp}.log"
    
    questions = load_questions(args.input_file)
    last_session_id = ""
    results = []

    for i, question in enumerate(questions, 1):
        print(f"\n{'='*60}")
        print(f"Question {i}/{len(questions)}: {question}")
        print('='*60)

        # Check if this is a continuation question (uses previous session)
        if question.startswith("CONTINUE:"):
            query = question[len("CONTINUE:"):].strip()
            session_id = last_session_id
            print(f"[Session continuation - using session_id: {session_id}]")
        else:
            query = question
            session_id = ""
            print("[New session]")

        result = invoke_question(query, session_id, headers)

        # Store session_id from response for potential follow-up questions
        if "session_id" in result:
            last_session_id = result["session_id"]

        print(json.dumps(result, indent=4))
        
        # Store result for output file
        results.append({
            "question_number": i,
            "question": query,
            "session_id": session_id if session_id else "new",
            "response": result
        })
    
    # Write all results to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print('='*60)


if __name__ == "__main__":
    main()
    # Example usage:
    # python invoke-question.py --input-file questions.md --email sameer.kumar@sita.aero