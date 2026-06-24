import json

def number_and_save_json(input_path, output_path):
    try:
        # Load the raw JSON data
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Process if the root is a list of items
        if isinstance(data, list):
            processed_data = []
            for index, item in enumerate(data, start=1):
                # Put the numeric ID at the very front of the dictionary
                new_item = {"id": f"{index}."}
                new_item.update(item)
                processed_data.append(new_item)
            
        # Process if the root is a single dictionary
        elif isinstance(data, dict):
            processed_data = {}
            for index, (key, value) in enumerate(data.items(), start=1):
                processed_data[f"{index}. {key}"] = value
        else:
            print("Error: Unsupported JSON root structure.")
            return

        # Write the formatted data out to the new file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
            
        print(f"Success! Numbered JSON saved to: {output_path}")
            
    except FileNotFoundError:
        print(f"Error: The file at '{input_path}' was not found.")
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON. Check if the file is well-formed.")

# Define your paths here
input_json_file = "../data/ChartQA_data/test/test_human_preprocessed.json"
output_json_file = "numbered_dataset.json"

number_and_save_json(input_json_file, output_json_file)