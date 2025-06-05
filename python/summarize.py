##### LIBRARIES #####
from datetime import datetime
from langchain.output_parsers.pydantic import PydanticOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from pydantic import BaseModel, Field, model_validator

import json
import os 
import random
import sys

##### TEST-LIBRARIES #####


##### INPUT VARIABLES SETTINGS #####
file_list_in_directory = [file for file in sorted(os.listdir(sys.argv[1]))]
random.seed(2024)

##### LLM PROMPTS #####
def create_prompt(format_instructions):
    QA_TEMPLATE = """
        {format_instructions}

        Read through all information: {data}
        Find me the Langitude, Longitude, Date of Issuance
        Provide me a maximum 500 words description
        For example:
            Longitude and latitude, UTM Coordinates: 43.79381, -80.386060
            {{"Langitude":"-80.386060","Longitude":"43.79381","Description":"A study on a waterbody"}}
    """
    return PromptTemplate(
        input_variables=["data"], 
        partial_variables={"format_instructions": format_instructions},
        template=QA_TEMPLATE)

# Define your desired data structure.
class Answer(BaseModel):
    Langitude: str = Field(description="langtitude of the location in the document")
    Longitude: str = Field(description="longitude of the location in the document")
    Description: str = Field(description="brief summary description of the document")
    Date_of_Issuance: str = Field(description="Dates of Issuance, usually found at the end of the document")


##### LLM VARIABLES SETTINGS #####
output_parser = JsonOutputParser(pydantic_object=Answer)
format_instructions = output_parser.get_format_instructions()
reasoning_model_list =["gemma3"]


##### FUNCTIONS #####
def import_txt_files(directory):
    library_of_files = {}
    for file in directory:
        uuid = file[:-4]
        file_address = os.path.join(sys.argv[1], file)
        with open(file_address, 'rt', encoding='utf-8') as text_file:
            library_of_files[uuid] = text_file.read()
    return library_of_files
    

def llm_summarize(data):
    summary = ''
    # print(input)
    sprompt = create_prompt(format_instructions)
    for reasoning_model in reasoning_model_list:
        sllm = OllamaLLM(model = reasoning_model, temperature = 0.0, format = 'json')
        summary_chain = sprompt | sllm | output_parser
        print(f"{reasoning_model} is summarizing:...", flush = True, end = '')
        ext_start_time = datetime.now()
        query_chain = summary_chain.invoke({"data": data})
        ext_end_time = datetime.now()
        seconds = (ext_end_time - ext_start_time).total_seconds()
        print(f"{seconds}", flush=True)
        summary = query_chain
    return summary


###---------------------------------------------------------------###
if __name__ == "__main__": 
    start_time = datetime.now()
    print("Running local at", start_time)
    dir_of_files = import_txt_files(file_list_in_directory)
    print(f"Finished {len(dir_of_files)} files")
    for doc_id, doc in dir_of_files.items():
        ocr_data = ''
        print(f"-----#####{doc_id}#####-----")
        # for doc_page in doc:
        # print(doc[:100])
            # ocr_data = list(set(ocr_data) | set(extracting_visual_pdf(doc_page)))
        # print("Current ocr_data:")
        # for data in ocr_data:
            # print("\t", data[:50], flush = True)
        # qa_result = QA_checker(prelim_result)
        # print("FINAL:\n\t", qa_result)
        # print(f"For ID {doc_id}, the content is:\n\t{doc}\n\tHeader is: {extracted_header}")
        # print(f"For ID {doc_id}, the content is:\n\t{doc}\n\t")
        # print(ocr_data)
        # input = ', '.join(ocr_data)
        summary = llm_summarize(doc)
        print(f"SUMMARY:\n\t{summary}")

        current_directory = os.getcwd()
        file_name = doc_id + '.json'
        target_directory = current_directory + '/json/'
        os.makedirs(target_directory, exist_ok = True)
        output_file = os.path.join(target_directory, file_name)
        json_data = json.dumps(summary, indent = 4)
        with open(output_file, "w") as ocr_result:
            ocr_result.write(json_data)
        print(f"{doc_id} written")

    end_time = datetime.now()
    seconds = (end_time - start_time).total_seconds()
    print(f"Total Execution time: {seconds} secs for {len(file_list_in_directory)} files at {end_time}", flush=True)
