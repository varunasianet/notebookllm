"""
utils.py

Functions:
- get_script: Get the dialogue from the LLM.
- call_llm: Call the LLM with the given prompt and dialogue format.
- get_audio: Get the audio from the TTS model from HF Spaces.
"""

import os   
import requests
import tempfile


import soundfile as sf
import spaces
import torch
from gradio_client import Client
from openai import OpenAI
from parler_tts import ParlerTTSForConditionalGeneration
from pydantic import ValidationError
from transformers import AutoTokenizer


MODEL_ID = "accounts/fireworks/models/llama-v3p1-405b-instruct"
JINA_URL = "https://r.jina.ai/"

client = OpenAI(
    base_url="https://api.fireworks.ai/inference/v1",
    api_key=os.getenv("FIREWORKS_API_KEY"),
)

hf_client = Client("mrfakename/MeloTTS")

# Initialize the model and tokenizer (do this outside the function for efficiency)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

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


def parse_url(url: str) -> str:
    """Parse the given URL and return the text content."""
    full_url = f"{JINA_URL}{url}"
    response = requests.get(full_url, timeout=60)
    return response.text

def generate_audio(text: str, speaker: str, language: str, voice: str) -> str:
    """Generate audio using the local Parler TTS model or HuggingFace client."""

    if language == "EN":
        # Adjust the description based on speaker and language
        if speaker == "Guest":
            description = f"{voice} has a slightly expressive and animated speech, speaking at a moderate speed with natural pitch variations. The voice is clear and close-up, as if recorded in a professional studio."
        else:  # host
            description = f"{voice} has a professional and engaging tone, speaking at a moderate to slightly faster pace. The voice is clear, warm, and sounds like a seasoned podcast host."

        # Prepare inputs
        input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
        prompt_input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

        # Generate audio
        generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        audio_arr = generation.cpu().numpy().squeeze()

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            sf.write(temp_file.name, audio_arr, model.config.sampling_rate, format='mp3')
        
        return temp_file.name
    
    else:
        accent = language
        if speaker == "Guest":
            speed = 0.9
        else:  # host
            speed = 1.1
        # Generate audio
        result = hf_client.predict(
            text=text, language=language, speaker=accent, speed=speed, api_name="/synthesize"
        )
        return result
