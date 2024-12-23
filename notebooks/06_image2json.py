import base64
import json
import os
import sys
from pathlib import Path

def image_to_json(image_path, message="This is an example message"):
    # Check if the file exists
    if not os.path.exists(image_path):
        print(f"Error: The file {image_path} does not exist.")
        return

    # Read the image file and encode it to base64
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    # Create the JSON object
    json_data = {
        "newMessage": message,
        "image": encoded_image
    }

    # Create the output JSON file path
    output_path = Path(image_path).with_suffix('.json')

    # Write the JSON data to the file
    with open(output_path, "w") as json_file:
        json.dump(json_data, json_file)

    print(f"JSON file created successfully: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_image_file> [optional_message]")
        sys.exit(1)

    image_path = sys.argv[1]
    message = sys.argv[2] if len(sys.argv) > 2 else "This is an example message"

    image_to_json(image_path, message)