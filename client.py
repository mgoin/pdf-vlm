#!/usr/bin/env python3
import argparse
import base64
import json
from io import BytesIO

from openai import OpenAI
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

def send_pixtral_request(client, images, model: str, temperature: float) -> dict:
    """
    Build and send a chat completion request to the Pixtral endpoint using the OpenAI client.
    The message payload includes a text prompt and one image (base64 data URL) per image.
    """
    # Define the prompt that instructs the model to output a JSON summary.
    prompt_text = (
        "Please provide a JSON summary of this section of the document. "
        "Describe the key points, visual elements, and any notable details. "
        "Respond only in valid JSON format."
    )

    # Build the content list: first the text prompt, then one entry per image.
    content = [{"type": "text", "text": prompt_text}]
    for image in images:
        data_url = pil_image_to_data_url(image)
        content.append({
            "type": "image_url",
            "image_url": {"url": data_url}
        })

    # Build the message payload.
    messages = [{
        "role": "user",
        "content": content
    }]

    # Send the request using OpenAI's ChatCompletion API.
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response

def main():
    parser = argparse.ArgumentParser(
        description="Convert a PDF into images and send them in groups to the Pixtral endpoint for JSON summarization."
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
        "--images_per_request",
        type=int,
        default=4,
        help="Number of images to send in each request (default: 4)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    args = parser.parse_args()

    client = OpenAI(
        api_key=args.token,
        base_url=args.endpoint,
    )

    try:
        # Convert the PDF into a list of PIL images (one per page).
        images = convert_from_path(args.pdf_path)
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return

    total_images = len(images)
    print(f"Converted PDF to {total_images} images.")

    # Process the images in groups (sections) of the specified size.
    for i in range(0, total_images, args.images_per_request):
        group = images[i:i + args.images_per_request]
        print(f"\nSending images {i + 1} to {i + len(group)} for summarization...")
        try:
            response = send_pixtral_request(
                client=client,
                images=group,
                model=args.model,
                temperature=args.temperature
            )
        except Exception as ex:
            print(f"Error during API call: {ex}")
            continue

        # Extract the response content.
        try:
            summary = response.choices[0].message.content
        except (KeyError, IndexError):
            summary = json.dumps(response, indent=2)
        print("JSON Summary for this section:")
        print(summary)

if __name__ == "__main__":
    main()

