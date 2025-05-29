##### LIBRARIES #####
from datetime import datetime
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers.pydantic import PydanticOutputParser
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaLLM
from io import BytesIO
from PIL import Image
from pydantic import BaseModel, Field, model_validator

import base64
import os 
# import shutil
import sys

##### TEST-LIBRARIES #####

##### INPUT VARIABLES SETTINGS #####
file_list_in_directory = [file for file in os.listdir(sys.argv[1]) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'))]

##### LLM PROMPTS #####
def create_prompt(format_instructions):
    QA_TEMPLATE = """
        READ THE CURRENT IMAGE ENCASED BY ``` AND FIND THE HEADERS FOR EACH SECTIONS:
        ```{document}```
        RETURN THE RESULT AS AN OBJECT WITH THOSE HEADERS ONLY.
        {format_instructions}
    """
    #Verify your answer, and if the result list has more than 2 items, then Value has multiple parts. Treat them all as one value only, and ignore the number in brackets in them. Retry to shorten it to format above.
    return PromptTemplate(
        input_variables=["document"], 
        partial_variables={"format_instructions": format_instructions},
        template=QA_TEMPLATE)


# Define your desired data structure.
class Docs(BaseModel):
    header: str = Field(description="Header of a section in the give file")
    body: str = Field(description="body of a section in the give file")

class QA_Answer(BaseModel):
    No_Special: str = Field(description="Existence of special characters in the answer")
    English: str = Field(description="Language of the answer")


##### LLM VARIABLES SETTINGS #####
model_select = 'llava'
output_parser = PydanticOutputParser(pydantic_object=Docs)  
format_instructions = output_parser.get_format_instructions()
llm = OllamaLLM(model = model_select, temperature = 0.0)
prompt = create_prompt(format_instructions)
llm_chain = prompt | llm | output_parser

vision_model_list = ["qwen2.5vl", "llama3.2-vision"]
reasoning_model_list =["llama3.1", "gemma3"]
##### FUNCTIONS #####
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

def llm_extract(text_content):
    print('INPUT:\n\t', text_content, flush = True, end = '\n\t')
    # LLM_start_time = datetime.now()
    # print(text_content, definition, example, flush = True)
    result = llm_chain.invoke({'document': text_content})
    return result

def intake_pdf_from_dir(directory):
    loaded_list_of_pdf_files = {}
    # Load the PDF file using pypdf library
    for pdf in directory:
        uuid = pdf[5:11]
        print(f"\tuuid: {uuid}")
        file_address = os.path.join(sys.argv[1], pdf)
        # pdf_file = pdfquery.PDFQuery(file_address)
        with open(file_address, 'rt', encoding='utf-8') as pdf_file:
            loaded_list_of_pdf_files[uuid] = pdf_file
    return loaded_list_of_pdf_files

def intake_img_from_dir(directory):
    loaded_list_of_img_files = {}

    # Load the PDF file using pypdf library
    for img in directory:
        uuid = img[-10:-4]
        print(f"\tuuid: {uuid}")
        file_address = os.path.join(sys.argv[1], img)
        pil_image = Image.open(file_address)
        image_b64 = convert_to_base64(pil_image)
        loaded_list_of_img_files[uuid] = file_address
    return loaded_list_of_img_files

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

def QA_prompt(data):
    text = data["text"]
    content_parts = []
    text_part = {"type": "text", "text": text}
    content_parts.append(text_part)
    return [HumanMessage(content=content_parts)]

def testing_visual_models(img):
    file_path = img
    pil_image = Image.open(file_path)
    image_b64 = convert_to_base64(pil_image)
    ocr_result = []
    for vision_model in vision_model_list:
        attempt = 1
        llm = ChatOllama(model=vision_model, temperature=0)
        chain = prompt_func | llm | StrOutputParser()
        while attempt < 10:
            query_chain = chain.invoke(
                {
                    "text": """
                        This form has a COMMENTS section to the bottom right. 
                        Identify the COMMENTS data and provide me only the comments. 
                        If it's not in english or include special characters, provide me the clean up version in english, without the special characters.
                        DO NOT INCLUDE ANY EXPLANATION
                        RETURN ONLY THE TEXT RESULT, NO SPECIAL CHARACTERS
                    """, 
                    "image": image_b64
                }
            )
            ocr_result.append(query_chain)
            attempt += 1
    return ocr_result

def QA_checker(text_input):
    qa_result = []
    for prelim_data in text_input:
        model_result = []
        for reasoning_model in reasoning_model_list:
            llm = ChatOllama(model=reasoning_model, temperature=0)
            chain = QA_prompt | llm | StrOutputParser()
            query_chain = chain.invoke({
                    "text": """
                        Is the result of ```{data}``` a comprehensible english phrase with no special characters?
                        If yes return 'Yes' else 'No'
                        DO NOT INCLUDE ANY EXPLANATION
                        RETURN ONLY THE TEXT RESULT, NO SPECIAL CHARACTERS, NO NEWLINE AT THE END OF YOUR ANSWERS
                    """,
                    "data": prelim_data
                })
            model_result.append(query_chain) ## ['yes']
        qa_result = list(set(qa_result) | set(model_result)) 
    return qa_result


###---------------------------------------------------------------###
if __name__ == "__main__": 
    start_time = datetime.now()
    print("Running local at", start_time)
    dir_of_imgs = intake_img_from_dir(file_list_in_directory)
    print(f"Finished {len(dir_of_imgs)} files")
    for doc_id, doc in dir_of_imgs.items():
        print(f"-----#####{doc_id}#####-----")
        # extracted_header = llm_extract(doc)
        prelim_result = testing_visual_models(doc)
        print("PRELIM:\n\t", prelim_result)
        qa_result = QA_checker(prelim_result)
        print("FINAL:\n\t", qa_result)
        # print(f"For ID {doc_id}, the content is:\n\t{doc}\n\tHeader is: {extracted_header}")
        # print(f"For ID {doc_id}, the content is:\n\t{doc}\n\t")

    end_time = datetime.now()
    seconds = (end_time - start_time).total_seconds()
    print(f"Total Execution time: {seconds} secs for {len(file_list_in_directory)} files at {end_time}", flush=True)