import base64
from io import BytesIO

# from IPython.display import HTML, display
from PIL import Image
import sys

def convert_to_base64(pil_image):
    """
    Convert PIL images to Base64 encoded strings

    :param pil_image: PIL image
    :return: Re-sized Base64 string
    """

    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")  # You can change the format if needed
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

# file_path = "pdf//images//20241211_114635.jpg"
# pil_image = Image.open(file_path)

# image_b64 = convert_to_base64(pil_image)
# plt_img_base64(image_b64)

# llama_chain = ["minicpm-v","llava","bakllava","llava-llama3","moondream","llava-phi3","granite3.2-vision","gemma3", "llama4", "qwen2.5vl", "llama3.2-vision"]
# llama_chain = ["minicpm-v","granite3.2-vision","gemma3", "qwen2.5vl", "llama3.2-vision"]
# llm = ChatOllama(model="qwen2.5vl", temperature=0)
# llm = ChatOllama(model="llama3.2-vision", temperature=0)
# attempt = 1
# for vision_model in llama_chain:

# while attempt < 10:
    # llm = ChatOllama(model=vision_model, temperature=0)

import base64

import httpx
from langchain.chat_models import init_chat_model

# Fetch PDF data
pdf_file_address = sys.argv[1]
def convert_to_base64(pdf_file_address):
    buffered = BytesIO()
    pdf_file_address.save(buffered, format="pdf")  # You can change the format if needed
    pdf_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return pdf_data

pdf_data = convert_to_base64(pdf_file_address)

# Pass to LLM
llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")

message = {
    "role": "user",
    "content": [
        {
            "type": "text",
            "text": "Describe the document:",
        },
        {
            "type": "file",
            "source_type": "base64",
            "data": pdf_data,
            "mime_type": "application/pdf",
        },
    ],
}
response = llm.invoke([message])
print(response.text())