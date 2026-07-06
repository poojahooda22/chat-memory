"""The one vision call per uploaded image.

One multimodal chat completion returns the photo's structured annotation: a caption plus the
atomic contexts a personal-memory query needs (people, pets, objects, environment, activity,
emotion, OCR text). Temporal and geographic context deliberately do NOT come from the model —
they come from EXIF (see exif.py); the model only reports what is visible.

Same JSON discipline as the rest of the pipeline: no response_format (the gateway rejects it
for some models), strict-JSON prompt + robust parse_json.
"""

import base64

from app.memory.prompts import parse_json

IMAGE_ANNOTATION_SYSTEM = """You annotate ONE image for a personal memory system.

Describe ONLY what is visible. Never guess who a person is, never invent a place name, never
assert anything the pixels do not show. People and pets are described, not identified — the
user attaches names later. Give each person/pet a confidence between 0 and 1.

Respond with strict JSON, exactly this shape:
{
  "caption": "one or two sentences describing the image",
  "kind": "photo" or "screenshot",
  "people": [{"description": "e.g. a woman in a red jacket", "confidence": 0.0}],
  "pets": [{"description": "e.g. a golden retriever", "species": "dog", "confidence": 0.0}],
  "objects": ["notable objects"],
  "environment": "e.g. rooftop terrace at sunset, or an app's chat screen",
  "activity": "what is happening, e.g. walking a dog",
  "emotion": "visible mood if any, else empty string",
  "ocr_text": "ALL readable text in the image, verbatim; empty string if none",
  "place_guess": "a place name ONLY if literally readable in the image (a sign, a caption), else null"
}"""


def annotate_image(client, model: str, image_bytes: bytes, content_type: str) -> dict:
    """One chat completion with the image attached as a data URL; returns the parsed dict."""
    data_url = f"data:{content_type};base64,{base64.b64encode(image_bytes).decode()}"
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": IMAGE_ANNOTATION_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Annotate this image."},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
    )
    return parse_json(response.choices[0].message.content)