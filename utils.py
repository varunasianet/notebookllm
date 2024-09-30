"""
utils.py

Functions:
- get_script: Get the dialogue from the LLM.
- call_llm: Call the LLM with the given prompt and dialogue format.
- get_audio: Get the audio from the TTS model from HF Spaces.
"""

import os
import requests
import time
from gradio_client import Client
from openai import OpenAI
from pydantic import ValidationError

from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav

MODEL_ID = "accounts/fireworks/models/llama-v3p1-405b-instruct"
JINA_URL = "https://r.jina.ai/"

client = OpenAI(
    base_url="https://api.fireworks.ai/inference/v1",
    api_key=os.getenv("FIREWORKS_API_KEY"),
)

hf_client = Client("mrfakename/MeloTTS")

# download and load all models
preload_models()


def generate_script(system_prompt: str, input_text: str, output_model):
    """Get the dialogue from the LLM."""
    # Load as python object
    try:
        response = call_llm(system_prompt, input_text, output_model)
        dialogue = output_model.model_validate_json(response.choices[0].message.content)
    except ValidationError as e:
        error_message = f"Failed to parse dialogue JSON: {e}"
        system_prompt_with_error = f"{system_prompt}\n\nPlease return a VALID JSON object. This was the earlier error: {error_message}"
        response = call_llm(system_prompt_with_error, input_text, output_model)
        dialogue = output_model.model_validate_json(response.choices[0].message.content)

    # Call the LLM again to improve the dialogue
    system_prompt_with_dialogue = f"{system_prompt}\n\nHere is the first draft of the dialogue you provided:\n\n{dialogue}."
    response = call_llm(
        system_prompt_with_dialogue, "Please improve the dialogue.", output_model
    )
    improved_dialogue = output_model.model_validate_json(
        response.choices[0].message.content
    )
    return improved_dialogue


def call_llm(system_prompt: str, text: str, dialogue_format):
    """Call the LLM with the given prompt and dialogue format."""
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        model=MODEL_ID,
        max_tokens=16_384,
        temperature=0.1,
        response_format={
            "type": "json_object",
            "schema": dialogue_format.model_json_schema(),
        },
    )
    return response


def parse_url(url: str) -> str:
    """Parse the given URL and return the text content."""
    full_url = f"{JINA_URL}{url}"
    response = requests.get(full_url, timeout=60)
    return response.text


def generate_podcast_audio(text: str, speaker: str, language: str, use_advanced_audio: bool) -> str:

    if use_advanced_audio:
        audio_array = generate_audio(text, history_prompt=f"v2/{language}_speaker_{'1' if speaker == 'Host (Jane)' else '3'}")

        file_path = f"audio_{language}_{speaker}.mp3"

        # save audio to disk
        write_wav(file_path, SAMPLE_RATE, audio_array)

        return file_path


    else:
        if speaker == "Guest":
            accent = "EN-US" if language == "EN" else language
            speed = 0.9
        else:  # host
            accent = "EN-Default" if language == "EN" else language
            speed = 1
        if language != "EN" and speaker != "Guest":
            speed = 1.1

        # Generate audio
        for attempt in range(3):
            try:
                result = hf_client.predict(
                    text=text,
                    language=language,
                    speaker=accent,
                    speed=speed,
                    api_name="/synthesize",
                )
                return result
            except Exception as e:
                if attempt == 2:  # Last attempt
                    raise  # Re-raise the last exception if all attempts fail
                time.sleep(1)  # Wait for 1 second before retrying
