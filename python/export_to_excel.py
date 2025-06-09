# import os
# import json
# import pandas as pd

# def convert_jsons_to_single_excel(folder_path, output_file):
#     all_data = []

#     for filename in os.listdir(folder_path):
#         if filename.lower().endswith(".json"):
#             json_path = os.path.join(folder_path, filename)
#             try:
#                 with open(json_path, "r", encoding="utf-8") as f:
#                     data = json.load(f)
#                     df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
#                     # Insert filename as first column
#                     df.insert(0, 'source_file', filename)
#                     all_data.append(df)
#             except Exception as e:
#                 print(f"❌ Failed to process {filename}: {e}")

#     if all_data:
#         combined_df = pd.concat(all_data, ignore_index=True)
#         combined_df.to_excel(output_file, index=False)
#         print(f"✅ Combined Excel written to: {output_file}")
#     else:
#         print("⚠️ No valid JSON files to combine.")

# # Example usage
# convert_jsons_to_single_excel("json/", "combined_excel_output.xlsx")

import os
import json
import pandas as pd

def convert_jsons_to_transposed_excel(folder_path, output_file):
    original_headers = set()
    records = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".json"):
            json_path = os.path.join(folder_path, filename)
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    entries = [data] if isinstance(data, dict) else data

                    for entry in entries:
                        record = {}
                        for key, value in entry.items():
                            original_headers.add(key)
                            record[f"{key}_Source"] = filename
                            record[f"{key}_Mined"] = value
                        records.append(record)
            except Exception as e:
                print(f"❌ Failed to process {filename}: {e}")

    if not records:
        print("⚠️ No valid JSON records found.")
        return

    # Build original header row (sorted for consistency)
    sorted_headers = sorted(original_headers)
    header_row = sorted_headers
    source_row = [f"{h}_Source" for h in sorted_headers]
    mined_row = [f"{h}_Mined" for h in sorted_headers]

    # Build table of rows (record-wise, each record = column)
    rows = [header_row, source_row, mined_row]
    for record in records:
        row = []
        for h in sorted_headers:
            row.append(record.get(f"{h}_Source", ""))
        for h in sorted_headers:
            row.append(record.get(f"{h}_Mined", ""))
        rows.append(row)

    # Transpose the full table: each column becomes a record
    df = pd.DataFrame(rows).transpose()
    df.columns = [f"Record_{i}" for i in range(df.shape[1])]
    df.insert(0, "Field", header_row + source_row + mined_row)

    df.to_excel(output_file, index=False)
    print(f"✅ Transposed Excel with original headers written to: {output_file}")

# Example usage
convert_jsons_to_transposed_excel("json/", "combined_transposed_with_headers.xlsx")
