"""
main.py
"""

# Standard library imports
import glob
import os
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Literal, Tuple

# Third-party imports
import gradio as gr
from loguru import logger
from pydantic import BaseModel
from pypdf import PdfReader
from pydub import AudioSegment

# Local imports
from prompts import SYSTEM_PROMPT
from utils import generate_script, generate_audio

class DialogueItem(BaseModel):
    """A single dialogue item."""

    speaker: Literal["Host (Jane)", "Guest"]
    text: str


class Dialogue(BaseModel):
    """The dialogue between the host and guest."""

    scratchpad: str
    name_of_guest: str
    dialogue: List[DialogueItem]


def generate_podcast(file: str) -> Tuple[str, str]:
    """Generate the audio and transcript from the PDF."""
    # Read the PDF file and extract text
    with Path(file).open("rb") as f:
        reader = PdfReader(f)
        text = "\n\n".join([page.extract_text() for page in reader.pages])

    # Call the LLM
    llm_output = generate_script(SYSTEM_PROMPT, text, Dialogue)
    logger.info(f"Generated dialogue: {llm_output}")

    # Process the dialogue
    audio_segments = []
    transcript = "" # start with an empty transcript
    total_characters = 0

    for line in llm_output.dialogue:
        logger.info(f"Generating audio for {line.speaker}: {line.text}")
        if line.speaker == "Host (Jane)":
            speaker = f"**Jane**: {line.text}"
        else:
            speaker = f"**{llm_output.name_of_guest}**: {line.text}"
        transcript += speaker + "\n\n"
        total_characters += len(line.text)

        # Get audio file path
        audio_file_path = generate_audio(line.text, line.speaker)
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
    description="Convert your PDFs into podcasts with open-source AI models.",
    fn=generate_podcast,
    inputs=[
        gr.File(
            label="PDF",
        ),
    ],
    outputs=[
        gr.Audio(label="Audio", format="mp3"),
        gr.Markdown(label="Transcript"),
    ],
    allow_flagging="never",
    api_name=False,
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch(show_api=False)
