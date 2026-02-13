import json
import sys
import re
import os
import numpy as np
from collections import Counter

# ─── Detección automática de dispositivo ─────────────────
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
MODEL_NAME = os.environ.get("VLLM_MODEL", "unsloth/Meta-Llama-3.1-8B-Instruct")
print(f"[ARCSolver] Backend detectado: {BACKEND} | Modelo: {MODEL_NAME}")

_model = None
_tokenizer = None

def _load_model():
    global _model, _tokenizer
    if _model is not None:
        return

    if BACKEND == "vllm":
        from vllm import LLM, SamplingParams
        gpu_mem = float(os.environ.get("VLLM_GPU_MEMORY_UTIL", "0.8"))
        max_len = int(os.environ.get("VLLM_MAX_MODEL_LEN", "12000"))
        dtype   = os.environ.get("VLLM_DTYPE", "half")
        print(f"[ARCSolver] Cargando vLLM en H200... {MODEL_NAME}")
        _model = LLM(
            model=MODEL_NAME,
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
        print(f"[ARCSolver] Cargando transformers en {device}...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map=device,
        )
        print(f"[ARCSolver] Modelo listo en {device}.")

    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("[ARCSolver] Cargando transformers en CPU (lento)...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
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
            outputs = _model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.95,
                pad_token_id=_tokenizer.eos_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return _tokenizer.decode(generated, skip_special_tokens=True).strip()


# ─── Utilidades de grid ───────────────────────────────────
def grid_to_str(grid):
    return "\n".join(" ".join(map(str, row)) for row in grid)

def grid_to_compact(grid):
    return "\n".join("".join(map(str, row)) for row in grid)

def grids_equal(a, b):
    return (np.array(a) == np.array(b)).all()

def majority_vote(candidates):
    if not candidates:
        return []
    serialized = [json.dumps(g, separators=(',', ':')) for g in candidates]
    most_common = Counter(serialized).most_common(1)[0][0]
    return json.loads(most_common)


# ─── Parser ───────────────────────────────────────────────
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
        nums_sep = re.findall(r'\d+', line)
        if nums_sep and all(len(n) == 1 for n in nums_sep):
            grid.append([int(n) for n in nums_sep])
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


# ─── Fallback heurístico ─────────────────────────────────
def heuristic_fallback(task, test_input):
    arr = np.array(test_input)
    candidates = [
        arr, np.rot90(arr,1), np.rot90(arr,2),
        np.rot90(arr,3), np.fliplr(arr), np.flipud(arr),
    ]
    train = task.get('train', [])
    if not train:
        return arr.tolist()
    out_shapes = set(
        (len(ex['output']), len(ex['output'][0]))
        for ex in train if ex.get('output')
    )
    expected_shape = list(out_shapes)[0] if len(out_shapes) == 1 else None
    for c in candidates:
        if expected_shape and c.shape == expected_shape:
            return c.tolist()
    return train[0]['output']


# ─── Prompt ──────────────────────────────────────────────
def build_prompt(task, test_input, compact=False):
    fmt = grid_to_compact if compact else grid_to_str
    system = (
        "You are an expert at solving ARC-AGI tasks.\n"
        "Study the input→output pattern from ALL training examples, then predict the output.\n"
        "Output ONLY the final grid as a Python list of lists of integers. Nothing else.\n"
    )
    examples_block = ""
    for i, ex in enumerate(task.get('train', []), 1):
        examples_block += (
            f"Example {i}:\nInput:\n{fmt(ex['input'])}\n"
            f"Output:\n{fmt(ex['output'])}\n\n"
        )
    test_block = (
        f"Test Input:\n{fmt(test_input)}\n\n"
        "Output grid (list of lists of ints ONLY):\n"
    )
    full_prompt = system + "\n" + examples_block + test_block
    if not compact and len(full_prompt) > 8000:
        return build_prompt(task, test_input, compact=True)
    return full_prompt


# ─── ARCSolver ───────────────────────────────────────────
class ARCSolver:
    def __init__(self, use_vllm=False, vllm_config=None):
        print(f"[ARCSolver] Inicializando con backend: {BACKEND}")

    def _solve_one(self, task, test_input):
        prompt = build_prompt(task, test_input)
        exp_rows = len(test_input)
        exp_cols = len(test_input[0]) if test_input else 0

        temperatures = [0.3, 0.6, 0.9]
        for attempt, temp in enumerate(temperatures):
            try:
                text = _generate(prompt, temperature=temp)
                grid = parse_grid_from_text(text, exp_rows, exp_cols)
                if grid:
                    print(f"  [✓] Intento {attempt+1} OK (temp={temp})")
                    return grid
                print(f"  [✗] Intento {attempt+1} sin grid válido (temp={temp})")
            except Exception as e:
                print(f"  [!] Error intento {attempt+1}: {e}")

        print("  [~] Usando fallback heurístico.")
        return heuristic_fallback(task, test_input)

    def solve(self, task):
        if not task or not task.get('test'):
            return []
        predictions = []
        for idx, test_case in enumerate(task.get('test', [])):
            print(f"[Solving test {idx+1}/{len(task['test'])}]")
            pred = self._solve_one(task, test_case['input'])
            predictions.append(pred)
        return predictions


# ─── Evaluación local ────────────────────────────────────
def evaluate(task, predictions):
    correct = sum(
        1 for pred, tc in zip(predictions, task.get('test', []))
        if 'output' in tc and grids_equal(pred, tc['output'])
    )
    total = len(predictions)
    return {"correct": correct, "total": total, "accuracy": correct/total if total else 0}


# ─── Entry point ─────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python arc_main.py task.json [--eval]")
        sys.exit(0)
    with open(sys.argv[1], 'r') as f:
        task = json.load(f)
    solver = ARCSolver()
    result = solver.solve(task)
    print("\n=== Predicciones ===")
    print(json.dumps(result, indent=2))
    if "--eval" in sys.argv:
        stats = evaluate(task, result)
        print(f"\n=== Evaluación: {stats['correct']}/{stats['total']} ({stats['accuracy']*100:.1f}%) ===")
