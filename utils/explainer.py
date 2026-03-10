from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import json
import os

model = "google/gemma-2-9b-it"
model = "meta-llama/Llama-3.1-8B-Instruct"

prompt_base = """
You are an Explainable AI (XAI) Analyst. Your task is to analyze a set of object detection results resulting of a reasoning pipeline that applies multiple heuristics to validate or filter base model predictions. You must compare these predictions against their ground truth and generate a concise, human-readable explanation of why the final pipeline decision was reached, especially when a conflict exists. A `null` ground truth or prediction indicates either a False Positive or a False Negative, respectively.

Input Data:
- A JSON array of detection records for a dataset. Each object contains the base model prediction, the ground truth, and the results of the multi-stage reasoning pipeline's heuristics. For each heuristic, you have 0 (failed) or 1 (passed) indicators. Some detections were not made by the base model and are estimated by the pipeline, these are tagged as "infered".

Objective:
1.  Identify Conflicts: Focus on cases where `predicited label` does not match `ground truth label` OR where a high-confidence prediction was filtered out.
2.  Generate Causal Narrative: Use the `h-results` to explain the *reason* for the final decision in terms of real-world logic, citing the most critical failed constraints.
3.  Summarize System Performance: For the entire set of input records, identify the most common reason for failure (e.g., "70% of false positives were eliminated by the temporal consistency check").

Output Format:
- Provide a structured Markdown output with two sections:

## Individual Detection Analysis
- [Detection ID]: [Final_Decision]. Reason: [1-2 sentence explanation of the most critical heuristic failure, referencing the details].

## Overall Scenario Summary
[2-3 sentence summary of the pipeline's performance on this batch, highlighting the most effective heuristic and the most common cause of initial false predictions or errors.]

Input JSON:
<json>{json}</json>

Markdown Output:
"""

with open('samples/pipeline_heuristics.json', 'r') as infile:
    data = json.load(infile)
selection = data[:len(data)//3]  # Use a subset to fit within token limits
prompt = prompt_base.format(json=json.dumps(selection))



tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)

print(len(prompt), len(selection))

sequences = pipeline(
    prompt,
    max_new_tokens=4096,
    do_sample=False,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)


for seq in sequences:
    print(f"{seq['generated_text']}")
