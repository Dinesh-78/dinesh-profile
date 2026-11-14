"""Clean, simple conversational interface for Dinesh's Career Profile."""

import sys
from pathlib import Path
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.guardrails.pipeline import get_pipeline

def main():
    st.set_page_config(
        page_title="Dinesh — AI Profile",
        page_icon="💼",
        layout="centered"
    )

    # Minimal Sidebar
    with st.sidebar:
        st.title("Veerapareddy Dinesh")
        st.caption("Software & Generative AI Engineer")
        st.markdown("📍 Bengaluru, India")

        st.divider()
        st.markdown("### 📬 Contact")
        st.markdown("- **Email:** [dineshveerapareddy@gmail.com](mailto:dineshveerapareddy@gmail.com)")
        st.markdown("- **Phone:** [+91-8374330906](tel:+918374330906)")
        st.markdown("- **LinkedIn:** [dinesh-veerapareddy](https://www.linkedin.com/in/dinesh-veerapareddy/)")
        st.markdown("- **Portfolio:** [veerapareddydinesh.cyou](https://www.veerapareddydinesh.cyou/)")
        st.markdown("- **GitHub:** [Dinesh-78](https://github.com/Dinesh-78)")
        st.markdown("- **Notice Period:** 15 days")

        st.divider()
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # Header
    st.title("Chat with Dinesh's Profile")
    st.caption("Ask anything about his work experience, tech stack, projects, or availability.")

    # Session state initialization
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Quick suggestion prompts (only shown when conversation hasn't started)
    selected_prompt = None
    if len(st.session_state.messages) == 0:
        st.markdown("##### Suggestions:")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("💼 Experience Overview", use_container_width=True):
                selected_prompt = "Can you give me an overview of Dinesh's work experience?"
        with col2:
            if st.button("🛠️ Tech Stack & Skills", use_container_width=True):
                selected_prompt = "What is Dinesh's core technical stack and skills?"
        with col3:
            if st.button("⏱️ Notice & Availability", use_container_width=True):
                selected_prompt = "What is Dinesh's notice period and current availability?"

    # Render existing chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    user_query = st.chat_input("Ask a question about Dinesh...")
    prompt = selected_prompt or user_query

    if prompt:
        # 1. Display user prompt
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Generate response via pipeline
        pipeline = get_pipeline()
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = pipeline.process(
                    query=prompt,
                    chat_history=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages[:-1]
                    ]
                )
                st.markdown(result.response)

        # 3. Save assistant response
        st.session_state.messages.append({"role": "assistant", "content": result.response})

if __name__ == "__main__":
    main()
