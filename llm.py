import os
import json
import time
from typing import List, Dict, Any
from PIL import Image
from google import genai
from google.genai import types

def extract_skills_with_gemini(question: Dict[str, Any], client: genai.Client) -> List[str]:
    text_content = question.get('text_content', '')
    options = question.get('options', [])
    image_url = question.get('image_url')

    group_text = question.get('group_text')
    group_image_url = question.get('group_image_url')

    options_text = ""
    if options and isinstance(options, list):
        options_text = "\n".join([f"{opt['label']}: {opt['text']}" for opt in options if isinstance(opt, dict) and 'label' in opt and 'text' in opt])

    # Construct the prompt dynamically to avoid excessive whitespace
    prompt_parts = ["Analysera följande fråga:"]

    if group_text:
        prompt_parts.append(f"Kontext:\n{group_text}")
        prompt_parts.append(f"Fråga:\n{text_content}")
    else:
        # If no separate context, just present the question directly to avoid redundancy
        prompt_parts.append(text_content)

    if options_text:
        prompt_parts.append(f"Alternativ:\n{options_text}")

    prompt = "\n\n".join(prompt_parts)
    
    contents = [prompt]
    
    # helper for loading images
    def load_image_if_exists(url_path):
        if url_path:
            potential_path = url_path.lstrip("/")
            if os.path.exists(potential_path):
                try:
                    img = Image.open(potential_path)
                    print(f"  Loaded image: {potential_path}")
                    return img
                except Exception as e:
                    print(f"  Error loading image {potential_path}: {e}")
        return None

    # Load group image first (e.g. for DTK charts)
    if group_image_url:
        img = load_image_if_exists(group_image_url)
        if img:
            contents.append(img)

    # Load specific question image
    if image_url:
        img = load_image_if_exists(image_url)
        if img:
            contents.append(img)

    # System instruction for the persona
    system_instruction = """
    Du är en expertpsykometriker specialiserad på det svenska högskoleprovet (SweSAT).
    Din uppgift är att analysera provfrågor och identifiera de specifika färdigheter (Knowledge Components) som krävs för att lösa dem.
    
    Tänk steg för steg:
    1.  Försök att lösa uppgiften själv för att förstå lösningsvägen.
    2.  Identifiera vilka begrepp, regler eller metoder som användes.
    3.  Abstrahera dessa till generella färdigheter (t.ex., "Substantivböjning" istället för "Ordet 'bilen'"). Beskriv färdigheterna med så få ord som möjligt. Bestäm max 5 färdigheter per fråga.
    
    Returnera ENDAST en JSON-lista med strängar som representerar färdigheterna. Inget annat.
    Exempel: ["Geometri", "Pythagoras sats", "Areaberäkning"]
    """

    retries = 0
    max_retries = 3
    base_delay = 2
    while retries <= max_retries:
        try:
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=list[str],
                    system_instruction=system_instruction,
                    # thinking_config stöds inte för gemini-2.5-pro just nu; ta bort eller använd en model som stöder det
                    thinking_config=types.ThinkingConfig(thinking_level="high"),
                    temperature=0.0 # for more deterministic results
                )
            )
            return json.loads(response.text)
        except Exception as e:
            print(f"  Error extracting skills for QID {question.get('id')}: {e}")
            if "429" in str(e) or "503" in str(e): # Rate limit or Service Unavailable
                if retries < max_retries:
                    delay = base_delay * (2 ** retries)
                    print(f"  Retrying in {delay} seconds...")
                    time.sleep(delay)
                    retries += 1
                else:
                    print("  Max retries reached.")
                    return []
            else:
                return []
    return []
