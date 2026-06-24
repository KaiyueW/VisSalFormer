def parse_results(file_path):
    """Parses the file and returns a dictionary mapping result_id to its correctness (True/False)"""
    results = {}
    current_id = None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('-----Result'):
                parts = line.split()
                if len(parts) > 1:
                    current_id = int(parts[1].strip(':'))
            
            if '-----Correct:' in line and current_id is not None:
                # Store True if 'True' is in the line, False if 'False' is in the line
                is_correct = 'True' in line
                results[current_id] = is_correct
                
    return results

# Replace with your actual file paths
file1_path = './qwen3vl_zeroshot_no_saliency.txt'
file2_path = './qwen3vl_zeroshot_with_saliency.txt'

# Extract results maps {id: True/False}
file1_results = parse_results(file1_path)
file2_results = parse_results(file2_path)

# Find IDs where File 2 is False but File 1 is True
targeted_failures = [
    res_id for res_id in file2_results 
    if not file2_results[res_id] and file1_results.get(res_id) == True
]

# Print the sorted result numbers
print(f"Total cases where File 2 is WRONG but File 1 is CORRECT: {len(targeted_failures)}")
print("Result numbers:")
print(sorted(targeted_failures))