
import json

file_path = "/Users/ellen-beatetysvaer/Documents/V24/FYS5429/LLM/testing-scripts/instruction-data.json"

with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


model_input = format_input(data[50])
desired_response = f"\n\n### Response:\n{data[50]['output']}"

print(model_input + desired_response)

