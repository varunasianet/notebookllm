"""
utils.py

Functions:
- get_script: Get the dialogue from the LLM.
- call_llm: Call the LLM with the given prompt and dialogue format.
- get_audio: Get the audio from the TTS model from HF Spaces.
"""

import os

from gradio_client import Client
from openai import OpenAI
from pydantic import ValidationError

MODEL_ID = "accounts/fireworks/models/llama-v3p1-405b-instruct"

client = OpenAI(
    base_url="https://api.fireworks.ai/inference/v1",
    api_key=os.getenv("FIREWORKS_API_KEY"),
)

hf_client = Client("mrfakename/MeloTTS")


def generate_script(system_prompt: str, input_text: str, output_model):
    """Get the dialogue from the LLM."""
    # Load as python object
    try:
        response = call_llm(system_prompt, input_text, output_model)
        dialogue = output_model.model_validate_json(
            response.choices[0].message.content
        )
    except ValidationError as e:
        error_message = f"Failed to parse dialogue JSON: {e}"
        system_prompt_with_error = f"{system_prompt}\n\nPlease return a VALID JSON object. This was the earlier error: {error_message}"
        response = call_llm(system_prompt_with_error, input_text, output_model)
        dialogue = output_model.model_validate_json(
            response.choices[0].message.content
        )
    return dialogue


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


def generate_audio(text: str, speaker: str) -> str:
    """Get the audio from the TTS model from HF Spaces."""
    if speaker == "Guest":
        accent = "EN-US"
        speed = 0.9
    else:  # host
        accent = "EN-Default"
        speed = 1
    result = hf_client.predict(
        text=text, language="EN", speaker=accent, speed=speed, api_name="/synthesize"
    )
    return result
