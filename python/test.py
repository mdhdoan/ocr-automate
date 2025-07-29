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


##### LLM PROMPTS #####
def create_prompt(format_instructions):
    QA_TEMPLATE = """
        {format_instructions}
     
    """
    return PromptTemplate(
        input_variables=[""], 
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
reasoning_model_list =[""] 


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
        print(query_chain, flush=True)
        summary = query_chain
    return summary

def run_live_loop():
    print("Program started. Type '!stop' to exit.")
    while True:
        user_input = input(">> ")
        if user_input.strip() == "!stop":
            print("Stopping program.")
            break
        # Do something with the input or run your logic
        print(f"You entered: {user_input}")


###---------------------------------------------------------------###
if __name__ == "__main__": 
    start_time = datetime.now()
    print("Importing data")
    end_time = datetime.now()
    seconds = (end_time - start_time).total_seconds()
    print(f"Total Execution time: {seconds} secs for {len(file_list_in_directory)} files at {end_time}", flush=True)
    run_live_loop()
