import os
import json
import pandas as pd

def convert_jsons_to_transposed_excel(folder_path, output_file):
    original_fields = set()
    records = []

    # Step 1: Load all data and gather all unique field names
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".json"):
            json_path = os.path.join(folder_path, filename)
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    entries = [data] if isinstance(data, dict) else data

                    for entry in entries:
                        original_fields.update(entry.keys())
                        records.append((filename, entry))
            except Exception as e:
                print(f"❌ Failed to process {filename}: {e}")

    if not records:
        print("⚠️ No valid JSON records found.")
        return

    sorted_fields = sorted(original_fields)

    # Step 2: Create header rows
    header_row = []
    source_row = []
    mined_row = []

    for field in sorted_fields:
        header_row.append(field)
        source_row.append(f"{field}_Source")
        mined_row.append(f"{field}_Mined")

    # Step 3: Create transposed table (rows = fields + _Source + _Mined; columns = records)
    field_names = []
    all_columns = []

    for i, (filename, data) in enumerate(records):
        column = []
        for field in sorted_fields:
            # Original header placeholder (1st row)
            column.append("")  # header row only in first row
        for field in sorted_fields:
            column.append(filename)  # _Source
        for field in sorted_fields:
            column.append(data.get(field, ""))  # _Mined
        all_columns.append(column)

    # Now build final transposed table
    field_names = header_row + source_row + mined_row
    df = pd.DataFrame(all_columns).transpose()
    df.insert(0, "Field", field_names)
    df.columns = ["Field"] + [f"Record_{i+1}" for i in range(len(records))]

    # Step 4: Save to Excel
    df.to_excel(output_file, index=False)
    print(f"✅ Transposed Excel with original headers written to: {output_file}")

# Example usage
convert_jsons_to_transposed_excel("json/", "combined_transposed_with_headers.xlsx")
