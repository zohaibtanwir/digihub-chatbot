"""
Simple Chat UI for DigiHub Chatbot Testing
Run with: streamlit run ui/app.py
"""

import streamlit as st
import requests
import json

# Configuration
API_URL = "http://localhost:8080/chatbot/v1/chat"
DEFAULT_HEADERS = {
    'x-digihub-emailid': 'sameer.kumar@sita.aero',
    'x-digihub-tenantid': '1',
    'x-digihub-traceid': '123456789',
    'authorization': 'Bearer '
}

st.set_page_config(page_title="DigiHub Chatbot Tester", page_icon="💬", layout="wide")

st.title("DigiHub Chatbot Tester")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    email = st.text_input("Email ID", value="sameer.kumar@sita.aero")
    tenant_id = st.text_input("Tenant ID", value="1")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.session_id = ""
        st.rerun()

    st.divider()
    st.caption("Session ID:")
    st.code(st.session_state.get("session_id", "Not started"))

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

# Chat input
if prompt := st.chat_input("Ask a question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                headers = {
                    'x-digihub-emailid': email,
                    'x-digihub-tenantid': tenant_id,
                    'x-digihub-traceid': '123456789',
                    'authorization': 'Bearer '
                }
                payload = {
                    "chat_session_id": st.session_state.session_id,
                    "query": prompt
                }

                response = requests.post(API_URL, json=payload, headers=headers, timeout=120)
                data = response.json()

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
