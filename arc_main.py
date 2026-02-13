import json
import sys
import re
import os
import argparse
import numpy as np
from collections import Counter

# ─── Paths del validador ──────────────────────────────────
MODELS_DIR = "/app/models"
MODEL_NAME = os.environ.get("VLLM_MODEL", "unsloth/Meta-Llama-3.1-8B-Instruct")

# ─── Detección de backend ─────────────────────────────────
def _detect_backend():
    try:
        from vllm import LLM, SamplingParams
        return "vllm"
    except ImportError:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"

BACKEND = _detect_backend()
print(f"[ARCSolver] Backend: {BACKEND} | Modelo: {MODEL_NAME}")

_model = None
_tokenizer = None

def _load_model():
    global _model, _tokenizer
    if _model is not None:
        return

    if BACKEND == "vllm":
        from vllm import LLM
        gpu_mem = float(os.environ.get("VLLM_GPU_MEMORY_UTIL", "0.8"))
        max_len = int(os.environ.get("VLLM_MAX_MODEL_LEN", "12000"))
        dtype   = os.environ.get("VLLM_DTYPE", "half")
        model_path = os.path.join(MODELS_DIR, MODEL_NAME.replace("/", "--"))
        load_from = model_path if os.path.exists(model_path) else MODEL_NAME
        print(f"[ARCSolver] Cargando vLLM desde: {load_from}")
        _model = LLM(
            model=load_from,
            dtype=dtype,
            gpu_memory_utilization=gpu_mem,
            max_model_len=max_len,
            trust_remote_code=True,
        )
        print("[ARCSolver] vLLM listo.")

    elif BACKEND in ("cuda", "mps"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        device = "cuda" if BACKEND == "cuda" else "mps"
        model_path = os.path.join(MODELS_DIR, MODEL_NAME.replace("/", "--"))
        load_from = model_path if os.path.exists(model_path) else MODEL_NAME
        print(f"[ARCSolver] Cargando transformers en {device} desde: {load_from}")
        _tokenizer = AutoTokenizer.from_pretrained(load_from)
        _model = AutoModelForCausalLM.from_pretrained(
            load_from,
            torch_dtype=torch.float16,
            device_map=device,
        )
        print(f"[ARCSolver] Modelo listo en {device}.")

    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_path = os.path.join(MODELS_DIR, MODEL_NAME.replace("/", "--"))
        load_from = model_path if os.path.exists(model_path) else MODEL_NAME
        print(f"[ARCSolver] Cargando en CPU desde: {load_from}")
        _tokenizer = AutoTokenizer.from_pretrained(load_from)
        _model = AutoModelForCausalLM.from_pretrained(load_from)
        print("[ARCSolver] Modelo listo en CPU.")


def _generate(prompt: str, temperature: float = 0.3, max_tokens: int = 512) -> str:
    _load_model()

    if BACKEND == "vllm":
        from vllm import SamplingParams
        params = SamplingParams(temperature=temperature, max_tokens=max_tokens, top_p=0.95)
        outputs = _model.generate([prompt], params)
        return outputs[0].outputs[0].text.strip()
    else:
        import torch
        device = "cuda" if BACKEND == "cuda" else ("mps" if BACKEND == "mps" else "cpu")
        inputs = _tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = _model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.95,
                pad_token_id=_tokenizer.eos_token_id,
            )
        generated = out[0][inputs["input_ids"].shape[1]:]
        return _tokenizer.decode(generated, skip_special_tokens=True).strip()


# ─── Utilidades de grid ───────────────────────────────────
def grid_to_str(grid):
    return "\n".join(" ".join(map(str, row)) for row in grid)

def grid_to_compact(grid):
    return "\n".join("".join(map(str, row)) for row in grid)

def grids_equal(a, b):
    return (np.array(a) == np.array(b)).all()

def parse_grid_from_text(text, expected_rows=None, expected_cols=None):
    matches = list(re.finditer(r'\[\s*\[[\d,\s\[\]]+\]\s*\]', text))
    for m in reversed(matches):
        try:
            candidate = json.loads(m.group())
            if _valid_grid(candidate, expected_rows, expected_cols):
                return candidate
        except Exception:
            pass

    code_block = re.search(r'```(?:python)?\s*([\s\S]*?)```', text)
    if code_block:
        try:
            candidate = json.loads(code_block.group(1).strip())
            if _valid_grid(candidate, expected_rows, expected_cols):
                return candidate
        except Exception:
            pass

    lines = text.strip().split('\n')
    grid = []
    for line in lines:
        nums = re.findall(r'\d+', line)
        if nums and all(len(n) == 1 for n in nums):
            grid.append([int(n) for n in nums])
        elif re.fullmatch(r'\d+', line.strip()):
            grid.append([int(c) for c in line.strip()])
    if grid and _valid_grid(grid, expected_rows, expected_cols):
        return grid
    return None

def _valid_grid(grid, expected_rows=None, expected_cols=None):
    if not grid or not isinstance(grid, list):
        return False
    if not all(isinstance(row, list) for row in grid):
        return False
    col_len = len(grid[0])
    if not all(len(row) == col_len for row in grid):
        return False
    if expected_rows and len(grid) != expected_rows:
        return False
    if expected_cols and col_len != expected_cols:
        return False
    return True

def heuristic_fallback(task, test_input):
    arr = np.array(test_input)
    candidates = [arr, np.rot90(arr,1), np.rot90(arr,2), np.rot90(arr,3), np.fliplr(arr), np.flipud(arr)]
    train = task.get('train', [])
    if not train:
        return arr.tolist()
    out_shapes = set((len(ex['output']), len(ex['output'][0])) for ex in train if ex.get('output'))
    expected_shape = list(out_shapes)[0] if len(out_shapes) == 1 else None
    for c in candidates:
        if expected_shape and c.shape == expected_shape:
            return c.tolist()
    return train[0]['output']

def build_prompt(task, test_input, compact=False):
    fmt = grid_to_compact if compact else grid_to_str
    system = (
        "You are an expert at solving ARC-AGI tasks.\n"
        "Study the input→output pattern from ALL training examples, then predict the output.\n"
        "Output ONLY the final grid as a Python list of lists of integers. Nothing else.\n"
    )
    examples_block = ""
    for i, ex in enumerate(task.get('train', []), 1):
        examples_block += f"Example {i}:\nInput:\n{fmt(ex['input'])}\nOutput:\n{fmt(ex['output'])}\n\n"
    test_block = f"Test Input:\n{fmt(test_input)}\n\nOutput grid (list of lists of ints ONLY):\n"
    full_prompt = system + "\n" + examples_block + test_block
    if not compact and len(full_prompt) > 8000:
        return build_prompt(task, test_input, compact=True)
    return full_prompt

def solve_task(task):
    predictions = []
    for idx, test_case in enumerate(task.get('test', [])):
        print(f"[Solving test {idx+1}/{len(task['test'])}]")
        test_input = test_case['input']
        exp_rows = len(test_input)
        exp_cols = len(test_input[0]) if test_input else 0
        prompt = build_prompt(task, test_input)
        pred = None
        for attempt, temp in enumerate([0.3, 0.6, 0.9]):
            try:
                text = _generate(prompt, temperature=temp)
                grid = parse_grid_from_text(text, exp_rows, exp_cols)
                if grid:
                    print(f"  [✓] Intento {attempt+1} OK (temp={temp})")
                    pred = grid
                    break
                print(f"  [✗] Intento {attempt+1} sin grid (temp={temp})")
            except Exception as e:
                print(f"  [!] Error intento {attempt+1}: {e}")
        if pred is None:
            print("  [~] Usando fallback heurístico.")
            pred = heuristic_fallback(task, test_input)
        predictions.append(pred)
    return predictions


# ─── FASE PREP: descargar modelo ─────────────────────────
def run_prep(input_dir, output_dir):
    print(f"[PREP] Descargando modelo {MODEL_NAME} en {MODELS_DIR}...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, MODEL_NAME.replace("/", "--"))

    if BACKEND == "vllm":
        # En vLLM basta con hacer snapshot_download
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=MODEL_NAME, local_dir=model_path)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        AutoTokenizer.from_pretrained(MODEL_NAME).save_pretrained(model_path)
        AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16
        ).save_pretrained(model_path)

    print(f"[PREP] Modelo guardado en {model_path}")


# ─── FASE INFERENCE: resolver puzzles ────────────────────
def run_inference(input_dir, output_dir):
    dataset_path = os.path.join(input_dir, "miner_current_dataset.json")
    output_path  = os.path.join(output_dir, "results.json")

    print(f"[INFERENCE] Leyendo dataset: {dataset_path}")
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    # dataset puede ser dict {task_id: task} o list
    if isinstance(dataset, dict):
        items = dataset.items()
    else:
        items = [(str(i), task) for i, task in enumerate(dataset)]

    for task_id, task in items:
        print(f"[INFERENCE] Resolviendo tarea: {task_id}")
        try:
            predictions = solve_task(task)
            results[task_id] = predictions
        except Exception as e:
            print(f"[INFERENCE] Error en tarea {task_id}: {e}")
            results[task_id] = []

    with open(output_path, 'w') as f:
        json.dump(results, f)

    print(f"[INFERENCE] Resultados guardados en {output_path}")
    print(f"[INFERENCE] Total tareas resueltas: {len(results)}")


# ─── Entry point ─────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["prep", "inference"], required=True)
    parser.add_argument("--input",  default="/input")
    parser.add_argument("--output", default="/output")
    args = parser.parse_args()

    if args.phase == "prep":
        run_prep(args.input, args.output)
    elif args.phase == "inference":
        run_inference(args.input, args.output)
```

Commit message:
```
fix: implement --phase prep/inference interface for Hone validator
