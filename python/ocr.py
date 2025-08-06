##### LIBRARIES #####
from datetime import datetime
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from io import BytesIO
from PIL import Image

import base64
import os 
import sys

##### TEST-LIBRARIES #####

##### INPUT VARIABLES SETTINGS #####
file_list_in_directory = [file for file in sorted(os.listdir(sys.argv[1])) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'))]

##### LLM PROMPTS #####
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


##### LLM VARIABLES SETTINGS #####
str_output_parser = StrOutputParser() 
vision_model_list = ["mistral-small3.2"]
# vision_model_list = ["minicpm-v","qwen2.5vl:32b", "gemma3:27b"]


##### FUNCTIONS #####
def convert_img_to_base64(pil_image):
    """
    Convert PIL images to Base64 encoded strings

    :param pil_image: PIL image
    :return: Re-sized Base64 string
    """

    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")  # You can change the format if needed
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def intake_pdf_from_dir(directory):
    pass

def intake_img_from_dir(directory):
    loaded_list_of_img_files = {}
    for img in directory:
        uuid = img[:13]
        print(f"\tuuid: {uuid}, {img[-10:-4]}")
        file_address = os.path.join(sys.argv[1], img)
        # print(f"\tfile_address: {file_address}")
        pil_image = Image.open(file_address)
        image_b64 = convert_img_to_base64(pil_image)
        if uuid not in loaded_list_of_img_files:
            loaded_list_of_img_files[uuid] = [image_b64]
        else:
            loaded_list_of_img_files[uuid].append(image_b64)
    return loaded_list_of_img_files

def testing_visual_models(image_b64):
    pass
    # # file_path = img
    # # pil_image = Image.open(file_path)
    # # image_b64 = convert_to_base64(pil_image)
    # ocr_result = []
    # for vision_model in vision_model_list:
    #     print(f"{vision_model} running", end = '', flush = True)
    #     attempt = 1
    #     llm = ChatOllama(model=vision_model, temperature=0)
    #     chain = prompt_func | llm | StrOutputParser()
    #     while attempt < 3:
    #         print('.', end = '', flush = True)
    #         query_chain = chain.invoke(
    #             {
    #                 "text": """
    #                     Read through all information and provide me a summary of them.
    #                 """, 
    #                 "image": image_b64
    #             }
    #         )
    #         ocr_result.append(query_chain)
    #         attempt += 1
    #     print(' :)')
    # return ocr_result

def extracting_visual_img(image_b64):
    ocr_result = ''
    for vision_model in vision_model_list:
        print(f"{vision_model} running...", flush = True, end = '')
        llm = ChatOllama(model=vision_model, temperature=0)
        chain = prompt_func | llm | str_output_parser
        ext_start_time = datetime.now()
        query_chain = chain.invoke(
            {
                "text": """
                    You are given an image of a scanned, typewritten document. Your task is to accurately transcribe the text from the image,
                    line by line. Read the entire document, do not skip any lines.
                    If a character is unreadable, represent it with "[unreadable]". 
                    Do not attempt to correct any spelling or grammatical errors or change any formatting
                    Prioritize accuracy over fluency.
                """, 
                "image": image_b64
            }
        )
        ext_end_time = datetime.now()
        seconds = (ext_end_time - ext_start_time).total_seconds()
        print(f"{seconds}", flush=True)
        print("\t", query_chain, flush = True)
        ocr_result = query_chain
    return ocr_result

def extracting_visual_pdf(pdf_file):
    pass
    # ocr_result = []
    # for vision_model in vision_model_list:
    #     print(f"{vision_model} running", flush = True)
    #     llm = ChatOllama(model=vision_model, temperature=0)
    #     chain = prompt_func | llm | StrOutputParser()
    #     query_chain = chain.invoke(
    #         {
    #             "text": """
    #                 Read through all information and extract the text as closely as possible.
    #             """, 
    #             "image": pdf_file
    #         }
    #     )
    #     # print("\t", query_chain[:50], flush = True)
    #     ocr_result.append(query_chain)
    # return ocr_result


###---------------------------------------------------------------###
if __name__ == "__main__": 
    start_time = datetime.now()
    print("Running local at", start_time)
    dir_of_files = intake_img_from_dir(file_list_in_directory)
    print(f"Finished {len(dir_of_files)} files")
    for doc_id, doc in dir_of_files.items():
        ocr_data = ''
        print(f"-----#####{doc_id}#####-----")
        for doc_page in doc:
            # print(doc_page)
            ocr_data = ocr_data + '' + extracting_visual_img(doc_page)

        current_directory = os.getcwd()
        file_name = doc_id + '.txt'
        target_directory = current_directory + '/txt/all_text/'
        os.makedirs(target_directory, exist_ok = True)
        output_file = os.path.join(target_directory, file_name)
        with open(output_file, "w+") as ocr_result:
            ocr_result.write(ocr_data)
        print(f"{file_name} written")

    end_time = datetime.now()
    seconds = (end_time - start_time).total_seconds()
    print(f"Total Execution time: {seconds} secs for {len(file_list_in_directory)} files at {end_time}", flush=True)
