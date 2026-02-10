"""
Simple Chat UI for DigiHub Chatbot Testing
Run with: streamlit run ui/app.py
"""

import streamlit as st
import requests
import json
import time

# Configuration
API_URL = "http://localhost:8080/chatbot/v1/chat"
STREAM_API_URL = "http://localhost:8080/chatbot/v1/chat/stream"
DEFAULT_HEADERS = {
    'x-digihub-emailid': 'sameer.kumar@sita.aero',
    'x-digihub-tenantid': '1',
    'x-digihub-traceid': '123456789',
    'authorization': 'Bearer '
}

st.set_page_config(page_title="DigiHub Chatbot Tester", page_icon="", layout="wide")

st.title("DigiHub Chatbot Tester")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    email = st.text_input("Email ID", value="sameer.kumar@sita.aero")
    tenant_id = st.text_input("Tenant ID", value="1")
    use_streaming = st.checkbox("Enable Streaming", value=True, help="Stream tokens as they arrive (faster perceived response)")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.session_id = ""
        st.rerun()

    st.divider()
    st.caption("Session ID:")
    st.code(st.session_state.get("session_id", "Not started"))

    st.divider()
    st.caption("Streaming Status:")
    st.info("Streaming: ON" if use_streaming else "Streaming: OFF")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = ""

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "metadata" in message:
            with st.expander("Details"):
                st.json(message["metadata"])


def stream_response(prompt, headers, session_id):
    """Stream response from the API using SSE."""
    payload = {
        "chat_session_id": session_id,
        "query": prompt
    }

    response = requests.post(
        STREAM_API_URL,
        json=payload,
        headers=headers,
        stream=True,
        timeout=120
    )

    collected_response = ""
    metadata = None
    new_session_id = session_id

    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                data = line[6:]  # Remove 'data: ' prefix

                if data == '[DONE]':
                    break

                try:
                    event = json.loads(data)
                    event_type = event.get('type')

                    if event_type == 'session':
                        new_session_id = event.get('session_id', session_id)
                    elif event_type == 'token':
                        token = event.get('content', '')
                        collected_response += token
                        yield {'type': 'token', 'content': collected_response}
                    elif event_type == 'metadata':
                        metadata = event.get('data', {})
                        yield {'type': 'metadata', 'data': metadata, 'session_id': new_session_id}
                    elif event_type == 'error':
                        yield {'type': 'error', 'message': event.get('message', 'Unknown error')}

                except json.JSONDecodeError:
                    continue

    # Yield final state if no metadata was received
    if metadata is None and collected_response:
        yield {
            'type': 'metadata',
            'data': {
                'response': collected_response,
                'confidence': 0,
                'score': 0,
                'citation': []
            },
            'session_id': new_session_id
        }


def non_streaming_response(prompt, headers, session_id):
    """Get response from the non-streaming API."""
    payload = {
        "chat_session_id": session_id,
        "query": prompt
    }

    response = requests.post(API_URL, json=payload, headers=headers, timeout=120)
    return response.json()


# Chat input
if prompt := st.chat_input("Ask a question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call API
    with st.chat_message("assistant"):
        headers = {
            'x-digihub-emailid': email,
            'x-digihub-tenantid': tenant_id,
            'x-digihub-traceid': '123456789',
            'authorization': 'Bearer '
        }

        if use_streaming:
            # Streaming mode
            try:
                # Show thinking indicator with elapsed time
                status_placeholder = st.empty()
                response_placeholder = st.empty()
                metadata = None
                final_response = ""
                first_token_received = False
                start_time = time.time()

                # Start with thinking spinner
                with status_placeholder.container():
                    st.markdown("⏳ _Thinking..._")

                for event in stream_response(prompt, headers, st.session_state.session_id):
                    if event['type'] == 'token':
                        if not first_token_received:
                            # First token - record elapsed time and update status
                            elapsed = time.time() - start_time
                            status_placeholder.caption(f"First token in {elapsed:.1f}s")
                            first_token_received = True
                        # Update the placeholder with accumulated response
                        response_placeholder.markdown(event['content'] + "▌")
                        final_response = event['content']
                    elif event['type'] == 'metadata':
                        metadata = event.get('data', {})
                        if 'session_id' in event:
                            st.session_state.session_id = event['session_id']
                        # Update with final formatted response
                        final_answer = metadata.get('response', final_response)
                        response_placeholder.markdown(final_answer)
                        final_response = final_answer
                        # Update status with total time
                        total_time = time.time() - start_time
                        status_placeholder.caption(f"Completed in {total_time:.1f}s")
                    elif event['type'] == 'error':
                        status_placeholder.empty()
                        st.error(event.get('message', 'Unknown error'))
                        final_response = event.get('message', 'Error occurred')
                    elif event['type'] == 'session':
                        # Session received, still waiting for tokens
                        elapsed = time.time() - start_time
                        with status_placeholder.container():
                            st.markdown(f"⏳ _Processing..._ ({elapsed:.1f}s)")

                # Show metadata
                if metadata:
                    with st.expander("Details"):
                        display_metadata = {
                            "confidence": metadata.get("confidence"),
                            "score": metadata.get("score"),
                            "citation": metadata.get("citation", []),
                            "disclaimer": metadata.get("disclaimer")
                        }
                        st.json(display_metadata)

                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_response,
                    "metadata": metadata if metadata else {}
                })

            except requests.exceptions.ConnectionError:
                error_msg = "Could not connect to API. Is the server running on localhost:8080?"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
        else:
            # Non-streaming mode (original behavior)
            with st.spinner("Thinking..."):
                try:
                    data = non_streaming_response(prompt, headers, st.session_state.session_id)

                    # Update session ID
                    if "session_id" in data:
                        st.session_state.session_id = data["session_id"]

                    # Display response
                    answer = data.get("response", "No response received")
                    st.markdown(answer)

                    # Show metadata
                    metadata = {
                        "confidence": data.get("confidence"),
                        "score": data.get("score"),
                        "citation": data.get("citation", []),
                        "disclaimer": data.get("disclaimer")
                    }
                    with st.expander("Details"):
                        st.json(metadata)

                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "metadata": metadata
                    })

                except requests.exceptions.ConnectionError:
                    error_msg = "Could not connect to API. Is the server running on localhost:8080?"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
