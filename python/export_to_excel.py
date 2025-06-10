import os
import json
import pandas as pd

def convert_jsons_to_single_excel(folder_path, output_file):
    all_data = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".json"):
            json_path = os.path.join(folder_path, filename)
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
                    # Insert filename as first column
                    df.insert(0, 'source_file', filename)
                    all_data.append(df)
            except Exception as e:
                print(f"❌ Failed to process {filename}: {e}")

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df.to_excel(output_file, index=False)
        print(f"✅ Combined Excel written to: {output_file}")
    else:
        print("⚠️ No valid JSON files to combine.")

# Example usage
convert_jsons_to_single_excel("json/", "combined_excel_output.xlsx")
