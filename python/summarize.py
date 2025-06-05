##### LIBRARIES #####
from datetime import datetime
from langchain.output_parsers.pydantic import PydanticOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from openpyxl import load_workbook, Workbook
from pydantic import BaseModel, Field, model_validator

import json
import os 
import random
import sys

##### TEST-LIBRARIES #####


##### INPUT VARIABLES SETTINGS #####
file_list_in_directory = [file for file in sorted(os.listdir(sys.argv[1]))]
random.seed(2024)
worksheet = load_workbook("DFO_PATH_IP.xlsx")['DFO_PATH_IP_Offset']

##### LLM PROMPTS #####
def create_prompt(format_instructions):
    QA_TEMPLATE = """
        {format_instructions}

        Read through all information:
        ```{data}```
        Latitude, Longitude, usually near the beginning of the document, starts with "Longitude and latitude, UTM Coordinates:". Some may include multiple coordinates, get them all
        Habitat Type mentioned in the document. "Fish habitat" is too generic.
        Fish_species are usually in section 4.
        Offset_footprint_size are usualy in section 4, documenting the footprint of each of the different offsetting measures and recording the measures according to the type of habitat they provide. Each offsetting type or location has its own line, so some projects have 2 or more lines of offsetting information
        Find me the Date of Issuance, usually at the end of the document.
        There are usually 4 conditions in each documents, all with varying contents
        Answer me in lowercase letters, if you have meter squared, use "m2" as the unit
        For example:
        Then fill in the schema below. Try to get as accurate as possible, even if the data type is not conventional.
            {{"Latitude":,"Longitude":,"Date_of_Issuance":, "Condition_summary_X":, "Habitat_Type":, "Fish_species":, "Offset_footprint_size":,
            "Vegetation_Cover":, "Boulder":, "Woody_coverage":, "Instream_structures":,}}
        If no data is found, try again one more time, then return "None" for that value
    """
    return PromptTemplate(
        input_variables=["data"], 
        partial_variables={"format_instructions": format_instructions},
        template=QA_TEMPLATE)

# Define your desired data structure.
# class Answer(BaseModel):
#     Langitude: str = Field(description="langtitude of the location in the document")
#     Longitude: str = Field(description="longitude of the location in the document")
#     Description: str = Field(description="brief summary description of the document")
#     Habitat_Type: str = Field(description="type of habitat described.")
#     Date_of_Issuance: str = Field(description="Dates of Issuance, usually found at the end of the document")


##### LLM VARIABLES SETTINGS #####
output_parser = JsonOutputParser()
format_instructions = output_parser.get_format_instructions()
reasoning_model_list =["qwq"]


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
