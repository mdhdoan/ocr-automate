import base64
from io import BytesIO

# from IPython.display import HTML, display
from PIL import Image


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


def plt_img_base64(img_base64):
    """
    Disply base64 encoded string as image

    :param img_base64:  Base64 string
    """
    # Create an HTML img tag with the base64 string as the source
    image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'
    # Display the image by rendering the HTML
    # display(HTML(image_html))


file_path = "pdf//images//20241211_114635.jpg"
pil_image = Image.open(file_path)

image_b64 = convert_to_base64(pil_image)
# plt_img_base64(image_b64)

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

# llama_chain = ["minicpm-v","llava","bakllava","llava-llama3","moondream","llava-phi3","granite3.2-vision","gemma3", "llama4", "qwen2.5vl", "llama3.2-vision"]
llama_chain = ["minicpm-v","granite3.2-vision","gemma3", "qwen2.5vl", "llama3.2-vision"]
# llm = ChatOllama(model="minicpm-v", temperature=0)
# llm = ChatOllama(model="moondream", temperature=0)
attempt = 1
for vision_model in llama_chain:

    while attempt < 10:
        llm = ChatOllama(model=vision_model, temperature=0)

        def prompt_func(data):
            text = data["text"]
            image = data["image"]

            image_part = {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{image}",
            }

            content_parts = []

            text_part = {"type": "text", "text": text}

            content_parts.append(image_part)
            content_parts.append(text_part)

            return [HumanMessage(content=content_parts)]


        from langchain_core.output_parsers import StrOutputParser

        chain = prompt_func | llm | StrOutputParser()

        query_chain = chain.invoke(
            {"text": "What is the handwritten message and what is the table's title?", "image": image_b64}
        )

        print("MODEL:", vision_model, "\n\tAttempt:", attempt,"\n\tANSWER:", query_chain)
        attempt += 1