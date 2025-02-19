#!/usr/bin/env python3
import argparse
import base64
import json
from io import BytesIO

import openai
from pdf2image import convert_from_path
from PIL import Image

def pil_image_to_data_url(image: Image.Image) -> str:
    """
    Convert a PIL image to a base64-encoded data URL in PNG format.
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/png;base64,{img_b64}"

def send_pixtral_request(client, images, model: str, temperature: float, prompt_text: str) -> dict:
    """
    Build and send a chat completion request with a text prompt and image inputs.
    """
    # Build the content list: first the text prompt, then one entry per image.
    content = [{"type": "text", "text": prompt_text}]
    for image in images:
        data_url = pil_image_to_data_url(image)
        content.append({
            "type": "image_url",
            "image_url": {"url": data_url}
        })
    messages = [{"role": "user", "content": content}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response

def main():
    parser = argparse.ArgumentParser(
        description="Convert a PDF into images, get preliminary JSON summaries per chunk, then aggregate into one final JSON summary."
    )
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file")
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:8000/v1",
        help="Pixtral endpoint base URL (default: http://localhost:8000/v1)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Pixtral-12B-2409",
        help="Model name (default: mistralai/Pixtral-12B-2409)"
    )
    parser.add_argument(
        "--token",
        type=str,
        default="dummy",
        help="Authorization token if required (default: dummy)"
    )
    parser.add_argument(
        "--images_per_chunk",
        type=int,
        default=4,
        help="Number of images (pages) per preliminary request (default: 4)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    args = parser.parse_args()

    client = openai.OpenAI(
        api_key=args.token,
        base_url=args.endpoint,
    )

    try:
        # Convert PDF into a list of images (one per page).
        images = convert_from_path(args.pdf_path)
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return

    total_pages = len(images)
    print(f"Converted PDF to {total_pages} pages.")

    chunk_summaries = []
    # Stage 1: Process PDF in chunks.
    preliminary_prompt = (
        "Extract and summarize the content of these pages. "
        "Return your answer as a JSON object that may include keys like 'title', 'authors', 'abstract', 'sections', "
        "and 'figures'. Do not worry about having all keys, but output valid JSON."
    )
    for i in range(0, total_pages, args.images_per_chunk):
        chunk = images[i:i + args.images_per_chunk]
        print(f"\nProcessing pages {i + 1} to {i + len(chunk)}...")
        try:
            response = send_pixtral_request(
                client=client,
                images=chunk,
                model=args.model,
                temperature=args.temperature,
                prompt_text=preliminary_prompt
            )
        except Exception as ex:
            print(f"Error during API call for chunk starting at page {i+1}: {ex}")
            continue

        try:
            chunk_summary = response.choices[0].message.content
        except (KeyError, IndexError):
            chunk_summary = json.dumps(response, indent=2)
        print("Chunk summary:")
        print(chunk_summary)
        chunk_summaries.append(chunk_summary)

    if not chunk_summaries:
        print("No summaries were produced from the PDF chunks.")
        return

    # Stage 2: Aggregate chunk summaries.
    aggregation_prompt = (
        "You are provided with several JSON summaries that each describe part of a document. "
        "Please combine them into one final, coherent JSON summary of the entire document. "
        "Your output must follow exactly this JSON schema:\n\n"
        '{\n'
        '  "title": "<document title or null>",\n'
        '  "authors": ["<author1>", "<author2>", ...],\n'
        '  "abstract": "<abstract text or null>",\n'
        '  "sections": [\n'
        '    { "title": "<section title or null>", "content": "<section summary>" },\n'
        '    ...\n'
        '  ],\n'
        '  "figures": [\n'
        '    { "caption": "<figure caption>" },\n'
        '    ...\n'
        '  ],\n'
        '  "conclusion": "<conclusion text or null>"\n'
        '}\n\n'
        "Merge the information carefully so that the final JSON is complete and coherent."
        "\nHere are the preliminary summaries:\n" + "\n---\n".join(chunk_summaries)
    )

    # In this stage, no images are needed. We send a text-only message.
    aggregation_messages = [{"role": "user", "content": aggregation_prompt}]
    print("\nAggregating chunk summaries into a final JSON summary...")
    try:
        agg_response = client.chat.completions.create(
            model=args.model,
            messages=aggregation_messages,
            temperature=args.temperature,
        )
    except Exception as ex:
        print(f"Error during aggregation API call: {ex}")
        return

    try:
        final_summary = agg_response.choices[0].message.content
    except (KeyError, IndexError):
        final_summary = json.dumps(agg_response, indent=2)
    print("\nFinal aggregated JSON summary:")
    print(final_summary)

if __name__ == "__main__":
    main()
