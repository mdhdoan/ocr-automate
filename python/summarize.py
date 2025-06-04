##### LIBRARIES #####
from datetime import datetime
from langchain_core.messages import HumanMessage
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers.pydantic import PydanticOutputParser
from langchain.output_parsers import RetryOutputParser
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaLLM
from pydantic import BaseModel, Field, model_validator

import json
import os 
# import shutil
import sys

##### TEST-LIBRARIES #####


##### INPUT VARIABLES SETTINGS #####
file_list_in_directory = [file for file in sorted(os.listdir(sys.argv[1]))]


##### LLM PROMPTS #####
def create_prompt(format_instructions):
    QA_TEMPLATE = """
        {format_instructions}

        Read through all information: {data}
        Then fill out the pydantic model given
    """
    return PromptTemplate(
        input_variables=["data"], 
        partial_variables={"format_instructions": format_instructions},
        template=QA_TEMPLATE)


# Define your desired data structure.
class Answer(BaseModel):
    No_Special: bool = Field(description="Existence of special characters in the given data")
    English: bool = Field(description="Language of the given data")
    Total_footprint_size: str = Field(description="Area size of activity, unit in meter squared")
    Habitat_type: str = Field(description="Type of habitat")
    Fish_type: str = Field(description="type of fish involved")
    Fish_stage: str = Field(description="stage of the fish maturity")
    Benefits: str = Field(description="Benefit of activity")
    Offsetting: str = Field(description="The offset activity described in the given data")
    Offsettingl_footprint_size: str = Field(description="The footprint size noted in the offset activities described in the given data")
    Location: str = Field(description="The location of this activity")
    Langitude: float = Field(description="langtitude of the location")
    Longitude: float = Field(description="longitude of the location")
    Vegetation: str = Field(description="Vegetation cover in the area of activity")
    Boulder: str = Field(description="Boulders and other stone cover in the area of activity")
    Aquatic_structure: str = Field(description="aquatic sturcture mentioned, such as bank, river, depth or anything else")
    Habitat_structure: str = Field(description="habitat of fish here")
    Habitat_other_structure: str = Field(description="habitat of non-fish here")
    Water_inspection: str = Field(description="Water inspection of this area")
    Fish_target: str = Field(description="Species of fish targeted")


##### LLM VARIABLES SETTINGS #####
output_parser = PydanticOutputParser(pydantic_object=Answer)
str_output_parser = StrOutputParser() 
format_instructions = output_parser.get_format_instructions()
reasoning_model_list =["llama3.1"]
##### FUNCTIONS #####
def import_txt_files(directory):
    library_of_files = {}
    for file in directory:
        uuid = file[:-4]
        with open(file, 'rt', encoding='utf-8') as text_file:
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
        # extracted_header = llm_extract(doc)
        # prelim_result = testing_visual_models(doc)
        # print("PRELIM:\n\t", prelim_result)
        # doc = ', '.join(doc)
        # for doc_page in doc:
        print(doc[:100])
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
        # summary = llm_summarize(ocr_data)
        # print(f"SUMMARY:\n\t{summary}")

        # current_directory = os.getcwd()
        # file_name = 'OCR_' + doc_id + '.json'
        # target_directory = current_directory + '/json/'
        # os.makedirs(target_directory, exist_ok = True)
        # output_file = os.path.join(target_directory, file_name)
        # json_data = json.dumps(summary, indent = 4)
        # with open(output_file, "w") as ocr_result:
        #     ocr_result.write(json_data)
        # print(f"{doc_id} written")

    end_time = datetime.now()
    seconds = (end_time - start_time).total_seconds()
    print(f"Total Execution time: {seconds} secs for {len(file_list_in_directory)} files at {end_time}", flush=True)
