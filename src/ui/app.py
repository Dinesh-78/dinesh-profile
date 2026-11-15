"""Clean, simple conversational interface with Live Voice Mode for Dinesh's Career Profile."""

import inspect
import sys
from pathlib import Path
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.guardrails.pipeline import get_pipeline
from src.ui.audio_handler import transcribe_audio, render_speech_synthesis

def main():
    st.set_page_config(
        page_title="Dinesh — AI Profile",
        page_icon="💼",
        layout="centered"
    )

    # Minimal Sidebar with Live Mode Controls
    with st.sidebar:
        st.title("Veerapareddy Dinesh")
        st.caption("Software & Generative AI Engineer")
        st.markdown("📍 Bengaluru, India")

        st.divider()
        st.markdown("### 🎙️ Live Voice Mode")
        live_voice = st.toggle(
            "Live Talk Mode",
            value=True,
            help="Talk directly using the mic beside the send button and hear voice responses."
        )
        auto_speak = st.toggle(
            "Auto-speak answers",
            value=True,
            disabled=not live_voice,
            help="Automatically play spoken responses in live mode."
        )

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
    if live_voice:
        st.caption("🎙️ **Live Voice Mode is active.** Type below or tap the microphone button beside the send button to talk!")
    else:
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
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Provide listen button on assistant responses in voice mode
            if message["role"] == "assistant" and live_voice:
                render_speech_synthesis(message["content"], auto_play=False)

    # Chat input with optional voice recording beside send button
    input_kwargs = {
        "placeholder": "Ask a question (or tap mic to talk)..." if live_voice else "Ask a question about Dinesh..."
    }
    # Check if accept_audio is supported in current Streamlit version
    sig = inspect.signature(st.chat_input)
    if "accept_audio" in sig.parameters and live_voice:
        input_kwargs["accept_audio"] = True

    user_submission = st.chat_input(**input_kwargs)

    prompt = selected_prompt
    is_voice_input = False

    # Process user submission
    if user_submission is not None and not prompt:
        # Check if submission is dict-like (ChatInputValue with text and/or audio)
        audio_file = getattr(user_submission, "audio", None)
        text_val = getattr(user_submission, "text", None)

        if audio_file is not None:
            with st.spinner("🎙️ Transcribing voice..."):
                try:
                    audio_bytes = audio_file.read()
                    transcribed = transcribe_audio(audio_bytes)
                    if transcribed:
                        prompt = transcribed
                        is_voice_input = True
                    else:
                        st.warning("Could not detect clear speech. Please tap the mic and speak clearly, or type your question.")
                except Exception as e:
                    st.error(f"Voice transcription error: {e}")
        elif text_val:
            prompt = text_val.strip()
        elif isinstance(user_submission, str):
            prompt = user_submission.strip()

    if prompt:
        # 1. Display user prompt
        display_text = f"🎙️ *\"{prompt}\"*" if is_voice_input else prompt
        st.session_state.messages.append({"role": "user", "content": display_text})
        with st.chat_message("user"):
            st.markdown(display_text)

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

                # Voice output in Live Mode
                if live_voice:
                    render_speech_synthesis(result.response, auto_play=auto_speak)

        # 3. Save assistant response
        st.session_state.messages.append({"role": "assistant", "content": result.response})

if __name__ == "__main__":
    main()
