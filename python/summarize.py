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
# worksheet = load_workbook("DFO_PATH_IP.xlsx")['DFO_PATH_IP_Offset']
golden_data_path = "json/golden.json"
# Habitat Type mentioned in the document. "Fish habitat" is too generic. Example: "Riverine"
#         Fish_species are usually in section 4.
#         Offset_footprint_size are usualy in section 4, documenting the footprint of each of the different offsetting measures and recording the measures according to the type of habitat they provide. 
#         Each offsetting type or location has its own line, so some projects have 2 or more lines of offsetting information. Give me the sum of all the numerical value of the areas in the section.
#         Offset area size are usually mentioned in section about "Conditions related to Offsetting".
#         For Instream_structures, here are some examples: "Pool/deepwater habitat", "Riffles (rivers)", "Undercut bank (e.g. lunker bunker)"
#         For Vegetation_Cover, here are some examples: "Emergent vegetation (e.g. cattails and rush)","Riparian vegetation (e.g. trees and shrubs and grass)"
#         There are usually 4 conditions in each documents, all with varying contents, usually numbered. 
#         Answer me in lowercase letters, if you have meter squared, use "m2" as the unit
#         "Condition_summary_X":, "Habitat_Type":, "Fish_species":, "Offset_footprint_size":,"Vegetation_Cover":, "Boulder":, "Woody_coverage":, "Instream_structures":,

##### LLM PROMPTS #####
def create_prompt(format_instructions):
    QA_TEMPLATE = """
        {format_instructions}

        Read through all information:
        ```{data}```
        Latitude, Longitude, usually near the beginning of the document, starts with "Longitude and latitude, UTM Coordinates:". Some may include multiple coordinates, get them all. Standard used is WGS 84. Given coordinates might be in UTM or NAD83, which you have to convert.
        Coordinates - a replacement in case no Latitude or Longitude can be specifically identified. Take all the content from the "Longitude and latitude, UTM Coordinates:" section
        Find me the Date of Issuance, usually at the end of the document. Format it as Mmm DD YYYY
        DO NOT REPEAT THIS DATA:
        ```{prev_data}```
        Then fill in the schema below. Try to get as accurate as possible, even if the data type is not conventional.
            {{"Latitude":,"Longitude":,"Coordinates":, "Date_of_Issuance":}}
        If no data is found, try again one more time, then return "None" for that value
        
    """
    return PromptTemplate(
        input_variables=["data", "prev_data"], 
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
reasoning_model_list =["qwq", "qwen3", "deepseek-r1:14b"]


##### FUNCTIONS #####
def import_txt_files(directory):
    library_of_files = {}
    for file in directory:
        uuid = file[:-4]
        file_address = os.path.join(sys.argv[1], file)
        with open(file_address, 'rt', encoding='utf-8') as text_file:
            library_of_files[uuid] = text_file.read()
    return library_of_files
    
def compare_json_values(json1, json2):
    keys_to_compare = ["Latitude", "Longitude", "Coordinates", "Date_of_Issuance"]
    mismatches = {}

    for key in keys_to_compare:
        if key not in json1 or key not in json2:
            mismatches[key] = {
                "json1": json1.get(key, "<missing>"),
                "json2": json2.get(key, "<missing>")
            }
            continue

        val1 = json1[key]
        val2 = json2[key]

        if key == "Date_of_Issuance":
            try:
                datetime.strptime(val1, "%b %d %Y")
                datetime.strptime(val2, "%b %d %Y")
                if val1 != val2:
                    mismatches[key] = {"json1": val1, "json2": val2}
            except (ValueError, TypeError):
                mismatches[key] = {
                    "json1": f"{val1} (invalid format)",
                    "json2": f"{val2} (invalid format)"
                }
        else:
            if val1 != val2:
                mismatches[key] = {"json1": val1, "json2": val2}

    return True if not mismatches else mismatches

def compare_with_golden(golden_path, record_key, json2):
    if not os.path.exists(golden_path):
        return {"error": f"File not found: {golden_path}"}

    with open(golden_path, 'r') as file:
        golden_data = json.load(file)

    golden_record = golden_data.get(record_key)
    if golden_record is None:
        return {"error": f"Key '{record_key}' not found in golden.json"}

    return compare_json_values(golden_record, json2)

def llm_summarize(data_id, data, prev_result):
    summary = ''
    # print(input)
    sprompt = create_prompt(format_instructions)
    for reasoning_model in reasoning_model_list:
        sllm = OllamaLLM(model = reasoning_model, temperature = 0.0, format = 'json')
        summary_chain = sprompt | sllm | output_parser
        print(f"{reasoning_model} is summarizing:...", flush = True, end = '')
        ext_start_time = datetime.now()
        query_chain = summary_chain.invoke({"data": data, "prev_data": prev_result})
        ext_end_time = datetime.now()
        seconds = (ext_end_time - ext_start_time).total_seconds()
        print(f"{seconds}", flush=True)
        print(query_chain, flush=True)
        comparison_result = compare_with_golden(golden_data_path, data_id[-5:], query_chain)
        summary = query_chain
        if comparison_result is True:
            summary = query_chain
        elif prev_result == query_chain:
            print("DUPLICATED\n\t", comparison_result)
            summary = query_chain
        else:
            print("NOT MATCHED\n\t", comparison_result)
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
        summary = llm_summarize(doc_id, doc, {})
        # print(f"SUMMARY:\n\t{summary}")

        # current_directory = os.getcwd()
        # file_name = doc_id + '.json'
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
