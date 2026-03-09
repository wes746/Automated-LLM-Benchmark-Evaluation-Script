import lm_eval
from lm_eval.models.huggingface import HFLM
import json

# --- CONFIGURATION ---
BASE_MODEL = "mistralai/Mistral-7B-v0.1"
PEFT_MODEL = "./qlora-finetuned-model" # Path to the LoRA adapters from the training script
TASKS = ["hellaswag", "arc_challenge", "truthfulqa_mc2"]
OUTPUT_FILE = "evaluation_results.json"

def run_evaluation():
    print(f"Loading base model: {BASE_MODEL}")
    print(f"Applying LoRA adapters: {PEFT_MODEL}")
    
    # 1. Initialize the model with LoRA adapters via the HuggingFace wrapper
    lm = HFLM(
        pretrained=BASE_MODEL,
        peft=PEFT_MODEL,
        device="cuda"
    )

    # 2. Execute the evaluation harness against selected benchmarks
    print(f"Running standardized benchmarks for tasks: {TASKS}")
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=TASKS,
        num_fewshot=0,
        batch_size="auto"
    )

    # 3. Save and output the metrics
    print(f"Evaluation complete. Saving comprehensive data to {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results["results"], f, indent=4)
    
    # Display core accuracy metrics to the terminal
    print("\n--- CORE METRICS ---")
    for task, metrics in results["results"].items():
        print(f"\n{task.upper()}:")
        for metric_name, value in metrics.items():
            if metric_name.endswith(",none"): 
                 print(f"  {metric_name.split(',')[0]}: {value:.4f}")
            elif type(value) is float:
                 print(f"  {metric_name}: {value:.4f}")

if __name__ == "__main__":
    run_evaluation()
