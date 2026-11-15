"""Unit tests for audio handling and voice mode."""

from src.ui.audio_handler import clean_text_for_speech, transcribe_audio

def test_clean_text_for_speech():
    raw_markdown = """
    ### Experience & Skills
    - Worked at **Carelon Global Solutions** as Senior Software Engineer.
    - Check [portfolio](https://veerapareddydinesh.cyou) for details.
    ```python
    code = "hidden"
    ```
    Notice period: `15 days`.
    """
    cleaned = clean_text_for_speech(raw_markdown)
    assert "###" not in cleaned
    assert "**" not in cleaned
    assert "https://" not in cleaned
    assert "code = \"hidden\"" not in cleaned
    assert "Carelon Global Solutions" in cleaned
    assert "portfolio" in cleaned
    assert "15 days" in cleaned

def test_transcribe_audio_empty_bytes():
    assert transcribe_audio(b"") == ""
    assert transcribe_audio(b"short") == ""
