import textwrap
import os
from dotenv import load_dotenv
from IPython.display import display
from IPython.display import Markdown
import PIL.Image
import requests
from io import BytesIO
import argparse
import base64
from openai import AzureOpenAI

# arguments

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

def load_image_from_path(image_path):
    """Load an image from a file path"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    return PIL.Image.open(image_path)

def load_image_from_url(image_url):
    """Load an image from a URL"""
    response = requests.get(image_url)
    response.raise_for_status()
    return PIL.Image.open(BytesIO(response.content))

def encode_image_to_base64(image_path):
    """Encode image to base64 for Azure OpenAI API"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

load_dotenv()
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-4o')

client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2024-02-15-preview",
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

def analyze_image_with_gpt5(image_path, prompt="Please explain what you see in this image in detail."):
    """Analyze an image using GPT-4o via Azure OpenAI"""
    try:
        # Encode the image
        base64_image = encode_image_to_base64(image_path)
        
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

def analyze_image_pair_with_gpt5(before_image_path, after_image_path, prompt=None):
    """Analyze a pair of before/after images using GPT-4o via Azure OpenAI"""
    if prompt is None:
        prompt = (
            "You are a visual-language model assistant. Analyze the BEFORE image (first image) and return a JSON object (no prose) with the following structure: \n"
            "For each food category (fruit, whole_grain, vegetable, sugary_bev, other), provide:\n"
            "  - \"items\": an array of bounding boxes, where each bounding box is [x_min, y_min, x_max, y_max] in normalized coordinates (0.0 to 1.0)\n"
            "    indicating where each item of that category is located in the BEFORE image.\n"
            "  If no items of a category are present, use an empty array [].\n"
            "  If there are multiple items of the same category (e.g., 2 fruits), provide multiple bounding boxes.\n\n"
            "Return a single JSON object with this structure. Example:\n"
            "{\n"
            "  \"fruit\": {\"items\": [[0.1, 0.2, 0.3, 0.4], [0.15, 0.25, 0.35, 0.45]]},\n"
            "  \"whole_grain\": {\"items\": [[0.4, 0.3, 0.6, 0.5]]},\n"
            "  \"vegetable\": {\"items\": [[0.2, 0.6, 0.4, 0.8]]},\n"
            "  \"sugary_bev\": {\"items\": [[0.7, 0.1, 0.9, 0.3]]},\n"
            "  \"other\": {\"items\": []}\n"
            "}\n"
            "Do NOT include any text outside the JSON object. If you cannot determine items for a category, use an empty array."
        )
    try:
        # Encode both images
        before_base64 = encode_image_to_base64(before_image_path)
        after_base64 = encode_image_to_base64(after_image_path)
        
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{before_base64}"
                            }
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{after_base64}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.0,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error analyzing image pair: {str(e)}"

def analyze_single_image_with_gpt5(image_path, prompt=None):
    """Analyze a single image with nutrition focus using GPT-4o via Azure OpenAI"""
    # Resize image to 512x512 before analysis
    image = PIL.Image.open(image_path)
    image = image.resize((512, 512))
    
    # Save resized image temporarily
    temp_path = image_path.replace('.jpg', '_temp_512.jpg').replace('.JPG', '_temp_512.jpg')
    image.save(temp_path)
    
    if prompt is None:
        prompt = (
           '''You are a visual-language model assistant. Analyze the meal image and return a JSON object (no prose) with the following structure:
For each food name (banana, strawberry, muffin, milk, etc.), provide:
  - "food_name": string,
  - "boundary_box": [x_min, y_min, x_max, y_max],  # coordinates of the bounding box'''
        )

    result = analyze_image_with_gpt5(temp_path, prompt)
    
    # Clean up temporary file
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Analyze school meal images using GPT-4o via Azure OpenAI')
    parser.add_argument('--image_path', type=str, help='Path to the image to analyze before A or B', required=True)
    args = parser.parse_args()

    # Check if required environment variables are set
    if not AZURE_OPENAI_API_KEY:
        print("Error: AZURE_OPENAI_API_KEY not found in environment variables")
        return
    if not AZURE_OPENAI_ENDPOINT:
        print("Error: AZURE_OPENAI_ENDPOINT not found in environment variables")
        return
    
    # Analyze a single image
    image_path = args.image_path
    print(f"Analyzing single image: {image_path}")
    
    print("=== SINGLE IMAGE ANALYSIS ===")
    analysis = analyze_single_image_with_gpt5(image_path)
    print(analysis)
    
    # save analysis to file
    with open(f"azure_gpt5o_analysis_{args.image_path.split('/')[-1]}.txt", "w") as f:
        f.write(analysis)

if __name__ == "__main__":
    main() 