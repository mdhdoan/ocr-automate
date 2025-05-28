##### LIBRARIES #####
from datetime import datetime
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers.pydantic import PydanticOutputParser
from langchain_ollama import OllamaLLM
from io import BytesIO
from PIL import Image
from pydantic import BaseModel, Field, model_validator

import base64
import os 
import pytesseract
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


##### LLM VARIABLES SETTINGS #####
model_select = 'llava'
output_parser = PydanticOutputParser(pydantic_object=Docs)  
format_instructions = output_parser.get_format_instructions()
llm = OllamaLLM(model = model_select, temperature = 0.0)
prompt = create_prompt(format_instructions)
llm_chain = prompt | llm | output_parser


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
        # image = Image.open(file_address)
        with open(file_address, 'rt', encoding='utf-8') as pdf_file:
            loaded_list_of_img_files[uuid] = pdf_file
        loaded_list_of_img_files[uuid] = file_address
    return loaded_list_of_img_files


###---------------------------------------------------------------###
if __name__ == "__main__": 
    start_time = datetime.now()
    print("Running local at", start_time)
    dir_of_jsons = intake_img_from_dir(file_list_in_directory)
    print(f"Finished {len(dir_of_jsons)} files")
    for doc_id, doc in dir_of_jsons.items():
        extracted_header = llm_extract(doc)
        # print(f"For ID {doc_id}, the content is:\n\t{doc}\n\tHeader is: {extracted_header}")
        print(f"For ID {doc_id}, the content is:\n\t{doc}\n\t")

    end_time = datetime.now()
    seconds = (end_time - start_time).total_seconds()
    print(f"Total Execution time: {seconds} secs for {len(file_list_in_directory)} files at {end_time}", flush=True)