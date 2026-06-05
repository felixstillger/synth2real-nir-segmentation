from google import genai
from google.api_core import retry
import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path='google_api.env')

client = genai.Client()
modelname = "gemini-2.5-flash"

# Configuration
selected_dir = r"C:\Users\benha\Downloads\selected_lora"
output_file = "captions_structured_gflash.json"
requests_per_minute = 9 #9
wait_time = 60 / requests_per_minute

prompt = "Caption the image."
# prompt = "Please provide a detailed caption for the image. We need " \
#          "high quality captions that are yet concise" \
#          "of the image content as well as object positions. The " \
#          "caption should be concise yet descriptive. " \
#          "Use keywords instead of full sentences separated by commas " \
#          "and limit the length to about 30-40 words. " \
#          "Do not include information on the mood or color spectrum." \
#          "Especially do not state anything about the color or that it" \
#          " is a grayscale image which is incorrect. Add a context descriptor" \
#          " like urban street scene or whatever you see in the image."

# 1st prompt attempt
# prompt = "The plan is to use this image for training a LoRA for the" \
#          "SDXL model. To get the best possible results, we need detailed, " \
#          "high quality captions that are yet concise. Please provide an adequate caption " \
#          "of the image content as well as object positions. The " \
#          "caption should be concise yet descriptive." \
#          "The attached image is a NIR image. " \
#          "Use simple language to describe the content and limit the length to a reasonable level."

# Define output structure
# image_caption = {
#     "properties": {
#         "setting": {
#             "type": "string",
#             "description": "General environment of the image (e.g., 'urban street', 'highway', 'rural road').",
#         },
#         "objects": {
#             "type": "array",
#             "items": {"type": "string", "maxChars": 10},
#             "description": "List of important visible objects (e.g., 'car', 'bus', 'traffic light').",
#         },
#         "relations": {
#             "type": "string",
#             "description": "Simple description of spatial relations (e.g., 'bus in front of car', 'pedestrian crossing road').",
#         },
#         # Group undesired information in these categories to filter them out later
#         "style": {
#             "type": "string",
#             "description": "Information regarding the color spectrum, e.g. grayscale, rgb, etc. and image style.",
#         },
#         "mood": {
#             "type": "string",
#             "description": "Information regarding the mood or atmosphere of the image."
#         }

#     },
#     "required": ["setting", "objects"]
# }
image_caption = {
  "type": "object",
  "properties": {
    "setting": {
      "type": "string",
      "description": "General environment (e.g., urban street, highway, driver-view), keep brief (<=25 chars)"
    },
    "objects": {
      "type": "array",
      "description": "List of visible objects (e.g., car, bus, person, bicycle, cyclist, tree, road, sidewalk)",
      "items": {
        "type": "string",
        "description": "Object name (e.g., car, tree), short"
      },
      "minItems": 1,
      "maxItems": 12
    },
    "relations": {
      "type": "string",
      "description": "Simple spatial relation (e.g., bus in front of car), brief (<=50 chars)"
    },
    "style": {
      "type": "string",
      "description": "Color/mode info (e.g., grayscale), optional, brief"
    },
    "mood": {
      "type": "string",
      "description": "Atmosphere/mood (e.g., calm, busy), optional, brief"
    }
  },
  "required": ["setting", "objects"],
  "propertyOrdering": ["setting", "objects", "relations", "style", "mood"]
}


def json_to_caption(data: dict, style_token="<nirstyle>") -> str:
    parts = []
    if "setting" in data and data["setting"]:
        parts.append(data["setting"])
    if "objects" in data and data["objects"]:
        parts.extend(data["objects"])
    if "relations" in data and data["relations"]:
        parts.append(data["relations"])
    # Join and add style token
    caption = f"{style_token} {', '.join(parts)}".lower()
    return caption

# Load existing progress
if os.path.exists(output_file):
    with open(output_file, 'r') as f:
        captions = json.load(f)
else:
    captions = {}

# Get all image files
image_files = []
for root, dirs, files in os.walk(selected_dir):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_files.append(os.path.join(root, file))

# Process images

for i, img_path in enumerate(image_files):
    img_name = os.path.relpath(img_path, selected_dir)

    # Skip if already processed
    if img_name in captions:
        print(f"Skipping {img_name} (already processed)")
        continue

    print(f"Processing {img_name} ({i+1}/{len(image_files)})")

    
    success = False
    retry_counter = 0
    while not success and retry_counter < 5:
        try:
            img = client.files.upload(file=img_path)
            response = client.models.generate_content(
                model=modelname,
                contents=[img, prompt],
                config={
                    'response_mime_type': 'application/json',
                    'response_json_schema': image_caption
                },
            )
            success = True
        except genai.errors.ServerError as e:
            print(f"Server error occurred: {e}")
            retry_counter += 1
            time.sleep(2**(retry_counter+1))  # Wait before retrying
            continue
        
        

    if not success:
        print(f"Failed to process {img_name} after 5 attempts")
        raise Exception(f"Failed to process {img_name} after 5 attempts")

    # Add nirstyle token and save
    # caption = "<nirstyle> " + response.text
    caption = json_to_caption(response.parsed)
    captions[img_name] = caption
    
    # Save progress after each successful request
    with open(output_file, 'w') as f:
        json.dump(captions, f, indent=2)
    
    print(f"Completed: {img_name}")
    
    # Rate limiting
    if i < len(image_files) - 1:  # Don't wait after last image
        time.sleep(wait_time)
            


print(f"Captioning complete. Results saved to {output_file}")
print(f"Total captions: {len(captions)}")