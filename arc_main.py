import json
import sys
import re
import os
import argparse
import numpy as np
from collections import Counter

# ─── Configuración y Rutas ────────────────────────────────
MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")
MODEL_NAME = os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")

# ─── Detección automática de acelerador ───────────────────
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

    model_path = os.path.join(MODELS_DIR, MODEL_NAME.replace("/", "--"))
    load_from = model_path if os.path.exists(model_path) else MODEL_NAME

    if BACKEND == "vllm":
        from vllm import LLM
        gpu_mem = float(os.environ.get("VLLM_GPU_MEMORY_UTIL", "0.85"))
        max_len = int(os.environ.get("VLLM_MAX_MODEL_LEN", "8192"))
        dtype = os.environ.get("VLLM_DTYPE", "auto")
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
        print(f"[ARCSolver] Cargando transformers en {device} desde: {load_from}")
        _tokenizer = AutoTokenizer.from_pretrained(load_from, trust_remote_code=True)
        _model = AutoModelForCausalLM.from_pretrained(
            load_from,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device,
            trust_remote_code=True,
        )
        print(f"[ARCSolver] Modelo listo en {device}.")

    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"[ARCSolver] Cargando transformers en CPU (lento) desde: {load_from}")
        _tokenizer = AutoTokenizer.from_pretrained(load_from, trust_remote_code=True)
        _model = AutoModelForCausalLM.from_pretrained(load_from, trust_remote_code=True)
        print("[ARCSolver] Modelo listo en CPU.")


def _generate(prompt: str, temperature: float = 0.2, max_tokens: int = 1024) -> str:
    _load_model()

    if BACKEND == "vllm":
        from vllm import SamplingParams
        params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.95,
        )
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
                temperature=temperature if temperature > 0 else 0.01,
                do_sample=temperature > 0,
                top_p=0.95,
                pad_token_id=_tokenizer.eos_token_id or _tokenizer.pad_token_id,
            )
        generated = out[0][inputs["input_ids"].shape[1]:]
        return _tokenizer.decode(generated, skip_special_tokens=True).strip()


# ─── Utilidades de Grid y Geometría ───────────────────────
def grid_to_str(grid):
    return "\n".join(" ".join(str(int(c)) for c in row) for row in grid)

def grid_to_compact(grid):
    return "\n".join("".join(str(int(c)) for c in row) for row in grid)

def grids_equal(a, b):
    try:
        return (np.array(a) == np.array(b)).all()
    except Exception:
        return False


def infer_output_shape(task, test_input):
    """
    Infiere dinámicamente el tamaño esperado de la cuadrícula de salida (filas, columnas).
    ARC tiene diferentes reglas: salida constante, igual a la entrada o factor de escala.
    """
    train = task.get('train', [])
    if not train:
        return len(test_input), (len(test_input[0]) if test_input else 0)

    # 1. ¿Todas las salidas de entrenamiento tienen exactamente el mismo tamaño fijo? (ej. 3x3)
    train_out_shapes = [
        (len(ex['output']), len(ex['output'][0]))
        for ex in train if ex.get('output') and len(ex['output']) > 0 and len(ex['output'][0]) > 0
    ]
    if train_out_shapes and len(set(train_out_shapes)) == 1:
        return train_out_shapes[0]

    # 2. ¿Cada ejemplo tiene siempre exactamente el mismo tamaño que su entrada?
    same_as_input = all(
        (len(ex['input']), len(ex['input'][0])) == (len(ex['output']), len(ex['output'][0]))
        for ex in train if ex.get('output') and ex.get('input') and len(ex['input']) > 0 and len(ex['output']) > 0
    )
    if same_as_input:
        return len(test_input), (len(test_input[0]) if test_input else 0)

    # 3. ¿Existe un ratio constante de escalado (ej. 2x, 3x, 0.5x)?
    ratios = set()
    for ex in train:
        if ex.get('input') and ex.get('output') and len(ex['input']) > 0 and len(ex['output']) > 0:
            in_r, in_c = len(ex['input']), len(ex['input'][0])
            out_r, out_c = len(ex['output']), len(ex['output'][0])
            if in_r > 0 and in_c > 0:
                ratios.add((out_r / in_r, out_c / in_c))

    if len(ratios) == 1:
        rf, cf = list(ratios)[0]
        return int(round(len(test_input) * rf)), int(round(len(test_input[0]) * cf))

    # Si no hay certeza determinista, permitimos dimensiones flexibles (None, None)
    return None, None


def _valid_grid(grid, expected_rows=None, expected_cols=None):
    """Valida que sea una lista rectangular de números enteros ARC [0-9]"""
    if not grid or not isinstance(grid, list):
        return False
    if not all(isinstance(row, list) for row in grid) or len(grid) == 0:
        return False

    col_len = len(grid[0])
    if col_len == 0:
        return False
    if not all(len(row) == col_len for row in grid):
        return False

    # Validar que todos los valores sean convertibles a enteros entre 0 y 9
    for row in grid:
        for val in row:
            if not isinstance(val, (int, float, np.integer)):
                return False
            if val < 0 or val > 9:
                return False

    # Validar dimensiones esperadas si están definidas estrictamente
    if expected_rows is not None and len(grid) != expected_rows:
        return False
    if expected_cols is not None and col_len != expected_cols:
        return False

    return True


def parse_grid_from_text(text, expected_rows=None, expected_cols=None):
    """Extrae la cuadrícula del texto del modelo mediante múltiples heurísticas defensivas"""
    if not text:
        return None

    # 1. Búsqueda de JSON bracket completo [[...]]
    matches = list(re.finditer(r'\[\s*\[[\d,\s\[\]]+\]\s*\]', text))
    for m in reversed(matches):
        try:
            candidate = json.loads(m.group())
            if _valid_grid(candidate, expected_rows, expected_cols):
                return [[int(c) for c in row] for row in candidate]
            # Si no encaja con las dimensiones pero no teníamos certeza absoluta, probar sin dimensiones
            if expected_rows is None and _valid_grid(candidate):
                return [[int(c) for c in row] for row in candidate]
        except Exception:
            pass

    # 2. Bloques de código Markdown ```python o ```json
    code_blocks = re.findall(r'```(?:python|json)?\s*([\s\S]*?)```', text)
    for block in reversed(code_blocks):
        try:
            candidate = json.loads(block.strip())
            if _valid_grid(candidate, expected_rows, expected_cols):
                return [[int(c) for c in row] for row in candidate]
        except Exception:
            pass

    # 3. Líneas de números en texto plano
    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    grid = []
    for line in lines:
        nums = re.findall(r'\b\d\b', line)
        if nums:
            grid.append([int(n) for n in nums])
        elif re.fullmatch(r'\d+', line):
            grid.append([int(c) for c in line])

    if grid and _valid_grid(grid, expected_rows, expected_cols):
        return grid

    # 4. Segundo pase flexible: si las dimensiones esperadas eran restrictivas
    if expected_rows is not None:
        for m in reversed(matches):
            try:
                candidate = json.loads(m.group())
                if _valid_grid(candidate):
                    return [[int(c) for c in row] for row in candidate]
            except Exception:
                pass

    return None


def heuristic_fallback(task, test_input, exp_rows=None, exp_cols=None):
    """Estrategia de salvaguarda inteligente si el LLM no produce salida válida"""
    arr = np.array(test_input)
    candidates = [
        arr,
        np.rot90(arr, 1),
        np.rot90(arr, 2),
        np.rot90(arr, 3),
        np.fliplr(arr),
        np.flipud(arr)
    ]

    # Si conocemos las dimensiones esperadas exactas
    if exp_rows is not None and exp_cols is not None:
        target_shape = (exp_rows, exp_cols)
        for c in candidates:
            if c.shape == target_shape:
                return c.tolist()

    # Si las salidas de train tienen un tamaño común
    train = task.get('train', [])
    if train:
        out_shapes = [(len(ex['output']), len(ex['output'][0])) for ex in train if ex.get('output')]
        if len(set(out_shapes)) == 1:
            shape = out_shapes[0]
            for c in candidates:
                if c.shape == shape:
                    return c.tolist()
            # Retornar el primer output de train si tiene la forma correcta
            if train[0].get('output') and len(train[0]['output']) == shape[0]:
                return train[0]['output']

    return arr.tolist()


# ─── Construcción del Prompt ARC ──────────────────────────
def build_prompt(task, test_input, exp_rows=None, exp_cols=None, compact=False):
    fmt = grid_to_compact if compact else grid_to_str

    system = (
        "You are the world's leading ARC-AGI reasoning engine.\n"
        "Your goal is to infer the core abstract transformation rule (geometry, symmetry, colors 0-9, movement, filling, or logic) "
        "and predict the test output grid.\n\n"
        "Rules:\n"
        "1. Colors: 0:black, 1:blue, 2:red, 3:green, 4:yellow, 5:grey, 6:magenta, 7:orange, 8:azure, 9:maroon.\n"
        "2. Output ONLY the predicted grid as a Python 2D array of integers, like [[0, 1], [1, 0]].\n"
        "3. Do not include explanations or markdown prose outside the grid.\n"
    )

    examples_block = ""
    for i, ex in enumerate(task.get('train', []), 1):
        examples_block += f"Example {i}:\nInput:\n{fmt(ex['input'])}\nOutput:\n{fmt(ex['output'])}\n\n"

    dim_hint = ""
    if exp_rows is not None and exp_cols is not None:
        dim_hint = f"Note: The expected output grid shape is exactly {exp_rows} rows by {exp_cols} columns.\n"

    test_block = f"Test Input:\n{fmt(test_input)}\n\n{dim_hint}Output grid (Python list of lists of integers ONLY):\n"

    full_prompt = system + "\n" + examples_block + test_block
    if not compact and len(full_prompt) > 7500:
        return build_prompt(task, test_input, exp_rows, exp_cols, compact=True)
    return full_prompt


# ─── Solver Core ──────────────────────────────────────────
def solve_task(task):
    predictions = []
    for idx, test_case in enumerate(task.get('test', [])):
        print(f"[Solving test {idx+1}/{len(task['test'])}]")
        test_input = test_case['input']

        # Deducir dinámicamente las dimensiones esperadas de la cuadrícula
        exp_rows, exp_cols = infer_output_shape(task, test_input)
        if exp_rows and exp_cols:
            print(f"  [Shape] Forma esperada deducida: {exp_rows}x{exp_cols}")
        else:
            print("  [Shape] Forma flexible / no fija.")

        prompt = build_prompt(task, test_input, exp_rows, exp_cols)
        pred = None

        # Reintentos con barrido de temperatura creciente
        temperatures = [0.1, 0.4, 0.8]
        for attempt, temp in enumerate(temperatures):
            try:
                text = _generate(prompt, temperature=temp)
                grid = parse_grid_from_text(text, exp_rows, exp_cols)
                if grid:
                    print(f"  [✓] Intento {attempt+1} exitoso (temp={temp}) - Dim: {len(grid)}x{len(grid[0])}")
                    pred = grid
                    break
                print(f"  [✗] Intento {attempt+1} sin grid válido (temp={temp})")
            except Exception as e:
                print(f"  [!] Error intento {attempt+1}: {e}")

        if pred is None:
            print("  [~] Aplicando fallback heurístico.")
            pred = heuristic_fallback(task, test_input, exp_rows, exp_cols)

        predictions.append(pred)
    return predictions


# ─── FASE PREP: Descargar modelo para Hone Validator ──────
def run_prep(input_dir, output_dir):
    print(f"[PREP] Descargando / Verificando modelo {MODEL_NAME} en {MODELS_DIR}...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, MODEL_NAME.replace("/", "--"))

    if BACKEND == "vllm":
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=MODEL_NAME, local_dir=model_path)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True).save_pretrained(model_path)
        AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        ).save_pretrained(model_path)

    print(f"[PREP] Modelo guardado y listo en {model_path}")


# ─── FASE INFERENCE: Resolver dataset de puzzles ──────────
def run_inference(input_dir, output_dir):
    dataset_path = os.path.join(input_dir, "miner_current_dataset.json")
    output_path = os.path.join(output_dir, "results.json")

    print(f"[INFERENCE] Leyendo dataset desde: {dataset_path}")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    if isinstance(dataset, dict):
        items = dataset.items()
    else:
        items = [(str(i), task) for i, task in enumerate(dataset)]

    for task_id, task in items:
        print(f"\n[INFERENCE] === Tarea: {task_id} ===")
        try:
            predictions = solve_task(task)
            results[task_id] = predictions
        except Exception as e:
            print(f"[INFERENCE] Error en tarea {task_id}: {e}")
            results[task_id] = []

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False)

    print(f"\n[INFERENCE] Resultados guardados en {output_path}")
    print(f"[INFERENCE] Total tareas procesadas: {len(results)}")


# ─── Evaluación Local Directa ─────────────────────────────
def evaluate(task, predictions):
    correct = sum(
        1 for pred, tc in zip(predictions, task.get('test', []))
        if 'output' in tc and grids_equal(pred, tc['output'])
    )
    total = len(predictions)
    return {"correct": correct, "total": total, "accuracy": correct/total if total else 0}


# ─── Entry point (Soporta Validator CLI y Local CLI) ─────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hone / ARC-AGI Solver")
    parser.add_argument("--phase", choices=["prep", "inference"], help="Fase del validador de Hone")
    parser.add_argument("--input", default="/input", help="Directorio de entrada del validador")
    parser.add_argument("--output", default="/output", help="Directorio de salida del validador")
    parser.add_argument("task_file", nargs="?", help="Ruta a un archivo task.json para pruebas locales")
    parser.add_argument("--eval", action="store_true", help="Evaluar precisión contra ground-truth")
    args = parser.parse_args()

    if args.phase == "prep":
        run_prep(args.input, args.output)
    elif args.phase == "inference":
        run_inference(args.input, args.output)
    elif args.task_file:
        with open(args.task_file, "r", encoding="utf-8") as f:
            task = json.load(f)
        preds = solve_task(task)
        print("\n=== Predicciones ===")
        print(json.dumps(preds, indent=2))
        if args.eval:
            stats = evaluate(task, preds)
            print(f"\n=== Evaluación: {stats['correct']}/{stats['total']} ({stats['accuracy']*100:.1f}%) ===")
    else:
        parser.print_help()
