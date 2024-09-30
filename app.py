"""
main.py
"""

# Standard library imports
import glob
import os
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Literal, Tuple, Optional

# Third-party imports
import gradio as gr
from loguru import logger
from pydantic import BaseModel, Field
from pypdf import PdfReader
from pydub import AudioSegment

# Local imports
from prompts import SYSTEM_PROMPT
from utils import generate_script, generate_podcast_audio, parse_url


class DialogueItem(BaseModel):
    """A single dialogue item."""

    speaker: Literal["Host (Jane)", "Guest"]
    text: str


class ShortDialogue(BaseModel):
    """The dialogue between the host and guest."""

    scratchpad: str
    name_of_guest: str
    dialogue: List[DialogueItem] = Field(..., description="A list of dialogue items, typically between 5 to 9 items")


class MediumDialogue(BaseModel):
    """The dialogue between the host and guest."""

    scratchpad: str
    name_of_guest: str
    dialogue: List[DialogueItem] = Field(..., description="A list of dialogue items, typically between 8 to 13 items")


LANGUAGE_MAPPING = {
    "English": "en",
    "Chinese": "zh",
    "French": "fr",
    "German": "de",
    "Hindi": "hi",
    "Italian": "it",
    "Japanese": "ja",
    "Korean": "ko",
    "Polish": "pl",
    "Portuguese": "pt",
    "Russian": "ru",
    "Spanish": "es",
    "Turkish": "tr"
}

MELO_TTS_LANGUAGE_MAPPING = {
    "en": "EN",
    "es": "ES",
    "fr": "FR",
    "zh": "ZJ",
    "ja": "JP",
    "ko": "KR",
}




def generate_podcast(
    files: List[str],
    url: Optional[str],
    question: Optional[str],
    tone: Optional[str],
    length: Optional[str],
    language: str,
    use_advanced_audio: bool,
) -> Tuple[str, str]:
    """Generate the audio and transcript from the PDFs and/or URL."""



    text = ""

    # Check if the selected language is supported by MeloTTS when not using advanced audio
    if not use_advanced_audio and language in ['German', 'Hindi', 'Italian', 'Polish', 'Portuguese', 'Russian', 'Turkish']:
        raise gr.Error(f"The selected language '{language}' is not supported without advanced audio generation. Please enable advanced audio generation or choose a supported language.")

    # Check if at least one input is provided
    if not files and not url:
        raise gr.Error("Please provide at least one PDF file or a URL.")

    # Process PDFs if any
    if files:
        for file in files:
            if not file.lower().endswith(".pdf"):
                raise gr.Error(
                    f"File {file} is not a PDF. Please upload only PDF files."
                )

            try:
                with Path(file).open("rb") as f:
                    reader = PdfReader(f)
                    text += "\n\n".join([page.extract_text() for page in reader.pages])
            except Exception as e:
                raise gr.Error(f"Error reading the PDF file {file}: {str(e)}")

    # Process URL if provided
    if url:
        try:
            url_text = parse_url(url)
            text += "\n\n" + url_text
        except ValueError as e:
            raise gr.Error(str(e))

    # Check total character count
    if len(text) > 100000:
        raise gr.Error(
            "The total content is too long. Please ensure the combined text from PDFs and URL is fewer than ~100,000 characters."
        )
    

    # Modify the system prompt based on the user input
    modified_system_prompt = SYSTEM_PROMPT
    if question:
        modified_system_prompt += f"\n\PLEASE ANSWER THE FOLLOWING QN: {question}"
    if tone:
        modified_system_prompt += f"\n\nTONE: The tone of the podcast should be {tone}."
    if length:
        length_instructions = {
            "Short (1-2 min)": "Keep the podcast brief, around 1-2 minutes long.",
            "Medium (3-5 min)": "Aim for a moderate length, about 3-5 minutes.",
        }
        modified_system_prompt += f"\n\nLENGTH: {length_instructions[length]}"
    if language:
        modified_system_prompt += (
            f"\n\nOUTPUT LANGUAGE <IMPORTANT>: The the podcast should be {language}."
        )

    # Call the LLM
    if length == "Short (1-2 min)":
        llm_output = generate_script(modified_system_prompt, text, ShortDialogue)
    else:
        llm_output = generate_script(modified_system_prompt, text, MediumDialogue)
    logger.info(f"Generated dialogue: {llm_output}")

    # Process the dialogue
    audio_segments = []
    transcript = ""
    total_characters = 0

    for line in llm_output.dialogue:
        logger.info(f"Generating audio for {line.speaker}: {line.text}")
        if line.speaker == "Host (Jane)":
            speaker = f"**Jane**: {line.text}"
        else:
            speaker = f"**{llm_output.name_of_guest}**: {line.text}"
        transcript += speaker + "\n\n"
        total_characters += len(line.text)

        language_for_tts = LANGUAGE_MAPPING[language]

        if not use_advanced_audio:
            language_for_tts = MELO_TTS_LANGUAGE_MAPPING[language_for_tts]

        # Get audio file path
        audio_file_path = generate_podcast_audio(
            line.text, line.speaker, language_for_tts, use_advanced_audio
        )
        # Read the audio file into an AudioSegment
        audio_segment = AudioSegment.from_file(audio_file_path)
        audio_segments.append(audio_segment)

    # Concatenate all audio segments
    combined_audio = sum(audio_segments)

    # Export the combined audio to a temporary file
    temporary_directory = "./gradio_cached_examples/tmp/"
    os.makedirs(temporary_directory, exist_ok=True)

    temporary_file = NamedTemporaryFile(
        dir=temporary_directory,
        delete=False,
        suffix=".mp3",
    )
    combined_audio.export(temporary_file.name, format="mp3")

    # Delete any files in the temp directory that end with .mp3 and are over a day old
    for file in glob.glob(f"{temporary_directory}*.mp3"):
        if os.path.isfile(file) and time.time() - os.path.getmtime(file) > 24 * 60 * 60:
            os.remove(file)

    logger.info(f"Generated {total_characters} characters of audio")

    return temporary_file.name, transcript


demo = gr.Interface(
    title="Open NotebookLM",
    description="""

<table style="border-collapse: collapse; border: none; padding: 20px;">
  <tr style="border: none;">
    <td style="border: none; vertical-align: top; padding-right: 30px; padding-left: 30px;">
      <img src="https://raw.githubusercontent.com/gabrielchua/daily-ai-papers/main/_includes/icon.png" alt="Open NotebookLM" width="120" style="margin-bottom: 10px;">
    </td>
    <td style="border: none; vertical-align: top; padding: 10px;">
      <p style="margin-bottom: 15px;"><strong>Convert</strong> your PDFs into podcasts with open-source AI models (Llama 3.1 405B and MeloTTS).</p>
      <p style="margin-top: 15px;">Note: Only the text content of the PDFs will be processed. Images and tables are not included. The total content should be no more than 100,000 characters due to the context length of Llama 3.1 405B.</p>
    </td>
  </tr>
</table>
""",
    fn=generate_podcast,
    inputs=[
        gr.File(
            label="1. 📄 Upload your PDF(s)", file_types=[".pdf"], file_count="multiple"
        ),
        gr.Textbox(
            label="2. 🔗 Paste a URL (optional)",
            placeholder="Enter a URL to include its content",
        ),
        gr.Textbox(label="3. 🤔 Do you have a specific question or topic in mind?"),
        gr.Dropdown(
            choices=["Fun", "Formal"],
            label="4. 🎭 Choose the tone",
            value="Fun"
        ),
        gr.Dropdown(
            choices=["Short (1-2 min)", "Medium (3-5 min)"],
            label="5. ⏱️ Choose the length",
            value="Medium (3-5 min)"
        ),
        gr.Dropdown(
            choices=list(LANGUAGE_MAPPING.keys()),
            value="English",
            label="6. 🌐 Choose the language"
        ),
        gr.Checkbox(
            label="7. 🔄 Use advanced audio generation? (Experimental)",
            value=False
        )
    ],
    outputs=[
        gr.Audio(label="Podcast", format="mp3"),
        gr.Markdown(label="Transcript"),
    ],
    allow_flagging="never",
    api_name="generate_podcast",
    theme=gr.themes.Soft(),
    concurrency_limit=3,
    examples=[
        [
            [str(Path("examples/1310.4546v1.pdf"))],
            "",
            "Explain this paper to me like I'm 5 years old",
            "Fun",
            "Short (1-2 min)",
            "English",
            True
        ],
        [
            [],
            "https://en.wikipedia.org/wiki/Hugging_Face",
            "How did Hugging Face become so successful?",
            "Fun",
            "Short (1-2 min)",
            "English",
            False
        ],
        [
            [],
            "https://simple.wikipedia.org/wiki/Taylor_Swift",
            "Why is Taylor Swift so popular?",
            "Fun",
            "Short (1-2 min)",
            "English",
            False
        ],
    ],
    cache_examples=True,
)

if __name__ == "__main__":
    demo.launch(show_api=True)
