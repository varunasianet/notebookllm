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
from pydantic import BaseModel
from pypdf import PdfReader
from pydub import AudioSegment

# Local imports
from prompts import SYSTEM_PROMPT
from utils import generate_script, generate_audio, parse_url


class DialogueItem(BaseModel):
    """A single dialogue item."""

    speaker: Literal["Host (Jenna)", "Guest"]
    text: str


class Dialogue(BaseModel):
    """The dialogue between the host and guest."""

    scratchpad: str
    name_of_guest: str
    dialogue: List[DialogueItem]


def generate_podcast(
    files: List[str],
    url: Optional[str],
    tone: Optional[str],
    voice: Optional[str],
    length: Optional[str],
    language: str
) -> Tuple[str, str]:
    """Generate the audio and transcript from the PDFs and/or URL."""
    print(tone, voice, length, language)
    text = ""

    # Change language to the appropriate code
    language_mapping = {
        "English": "EN",
        "Spanish": "ES",
        "French": "FR",
        "Chinese": "ZH",
        "Japanese": "JP",
        "Korean": "KR",
    }

    # Change voice to the appropriate code
    voice_mapping = {
        "Male": "Gary",
        "Female": "Laura",
    }

    # Check if at least one input is provided
    if not files and not url:
        raise gr.Error("Please provide at least one PDF file or a URL.")

    # Process PDFs if any
    if files:
        for file in files:
            if not file.lower().endswith('.pdf'):
                raise gr.Error(f"File {file} is not a PDF. Please upload only PDF files.")

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
        raise gr.Error("The total content is too long. Please ensure the combined text from PDFs and URL is fewer than ~100,000 characters.")
    
    # Modify the system prompt based on the chosen tone and length
    modified_system_prompt = SYSTEM_PROMPT
    if tone:
        modified_system_prompt += f"\n\nTONE: The tone of the podcast should be {tone}."
    if length:
        length_instructions = {
            "Short (1-2 min)": "Keep the podcast brief, around 1-2 minutes long.",
            "Medium (3-5 min)": "Aim for a moderate length, about 3-5 minutes.",
        }
        modified_system_prompt += f"\n\nLENGTH: {length_instructions[length]}"
    if language:
        modified_system_prompt += f"\n\nOUTPUT LANGUAGE <IMPORTANT>: The the podcast should be {language}."

    # Call the LLM
    llm_output = generate_script(modified_system_prompt, text, Dialogue)
    logger.info(f"Generated dialogue: {llm_output}")

    # Process the dialogue
    audio_segments = []
    transcript = ""
    total_characters = 0

    for line in llm_output.dialogue:
        print(line.speaker, line.text, language_mapping[language], voice_mapping[voice])
        logger.info(f"Generating audio for {line.speaker}: {line.text}")
        if line.speaker == "Host (Jenna)":
            speaker = f"**Jenna**: {line.text}"
        else:
            speaker = f"**{llm_output.name_of_guest}**: {line.text}"
        transcript += speaker + "\n\n"
        total_characters += len(line.text)

        # Get audio file path
        audio_file_path = generate_audio(line.text, line.speaker, language_mapping[language], voice_mapping[voice])
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
    description="Convert your PDFs into podcasts with open-source AI models (Llama 3.1 405B and MeloTTS). \n \n Note: Only the text content of the PDFs will be processed. Images and tables are not included. The total content should be no more than 100,000 characters due to the context length of Llama 3.1 405B.",
    fn=generate_podcast,
    inputs=[
        gr.File(
            label="1. 📄 Upload your PDF(s)",
            file_types=[".pdf"],
            file_count="multiple"
        ),
        gr.Textbox(
            label="2. 🔗 Paste a URL (optional)",
            placeholder="Enter a URL to include its content"
        ),
        gr.Radio(
            choices=["Fun", "Formal"],
            label="3. 🎭 Choose the tone",
            value="Fun"
        ),
        gr.Radio(
            choices=["Male", "Female"],
            label="4. 🎭 Choose the guest's voice",
            value="Female"
        ),
        gr.Radio(
            choices=["Short (1-2 min)", "Medium (3-5 min)"],
            label="5. ⏱️ Choose the length",
            value="Medium (3-5 min)"
        ),
        gr.Dropdown(
            choices=["English", "Spanish", "French", "Chinese", "Japanese", "Korean"],
            value="English",
            label="6. 🌐 Choose the language (Highly experimental, English is recommended)",
        ),
    ],
    outputs=[
        gr.Audio(label="Audio", format="mp3"),
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
            "Fun",
            "Female",
             "Medium (3-5 min)",
            "English"
        ],
        [
            [],
            "https://en.wikipedia.org/wiki/Hugging_Face",
            "Fun",
            "Male"
            "Short (1-2 min)",
            "English"
        ],
        [
            [],
            "https://simple.wikipedia.org/wiki/Taylor_Swift",
            "Fun",
            "Female"
            "Short (1-2 min)",
            "English"
        ],
    ],
    cache_examples=True,
)

if __name__ == "__main__":
    demo.launch(show_api=True)
