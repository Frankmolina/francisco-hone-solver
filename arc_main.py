import json
import sys
import re
import numpy as np
from collections import Counter
from vllm import LLM, SamplingParams

# ─────────────────────────────────────────────
# Configuración global
# ─────────────────────────────────────────────
DEFAULT_VLLM_CONFIG = {
    "model": "unsloth/Meta-Llama-3.1-8B-Instruct",
    "dtype": "half",
    "gpu_memory_utilization": 0.85,
    "max_model_len": 12000,
}

NUM_VOTES      = 3      # Muestras por test case para majority voting
MAX_RETRIES    = 2      # Reintentos si el parseo falla
TEMPERATURES   = [0.3, 0.6, 0.9]   # Escalada si los intentos anteriores fallan


# ─────────────────────────────────────────────
# Utilidades de grid
# ─────────────────────────────────────────────
def grid_to_str(grid: list) -> str:
    return "\n".join(" ".join(map(str, row)) for row in grid)


def grid_to_compact(grid: list) -> str:
    """Formato compacto sin espacios: ahorra tokens."""
    return "\n".join("".join(map(str, row)) for row in grid)


def grids_equal(a, b) -> bool:
    return (np.array(a) == np.array(b)).all()


def majority_vote(candidates: list) -> list:
    """Devuelve el grid más frecuente entre los candidatos."""
    if not candidates:
        return []
    serialized = [json.dumps(g, separators=(',', ':')) for g in candidates]
    most_common = Counter(serialized).most_common(1)[0][0]
    return json.loads(most_common)


# ─────────────────────────────────────────────
# Parser robusto del output del LLM
# ─────────────────────────────────────────────
def parse_grid_from_text(text: str, expected_rows: int = None, expected_cols: int = None):
    """
    Intenta extraer un grid (list of lists de ints) del texto generado.
    Estrategias en orden de prioridad:
      1. Último bloque [[...]] válido
      2. Bloque ```python / ``` con list of lists
      3. Líneas consecutivas de dígitos separados
    """
    # Estrategia 1: último bloque [[...]]
    matches = list(re.finditer(r'\[\s*\[[\d,\s\[\]]+\]\s*\]', text))
    for m in reversed(matches):
        try:
            candidate = json.loads(m.group())
            if _valid_grid(candidate, expected_rows, expected_cols):
                return candidate
        except Exception:
            pass

    # Estrategia 2: bloque de código
    code_block = re.search(r'```(?:python)?\s*([\s\S]*?)```', text)
    if code_block:
        try:
            candidate = json.loads(code_block.group(1).strip())
            if _valid_grid(candidate, expected_rows, expected_cols):
                return candidate
        except Exception:
            pass

    # Estrategia 3: líneas de dígitos (soporte dígitos separados o juntos)
    lines = text.strip().split('\n')
    grid = []
    for line in lines:
        # Acepta "1 2 3" o "123"
        nums_sep = re.findall(r'\d+', line)
        if nums_sep and all(len(n) == 1 for n in nums_sep):
            grid.append([int(n) for n in nums_sep])
        elif re.fullmatch(r'\d+', line.strip()):
            grid.append([int(c) for c in line.strip()])

    if grid and _valid_grid(grid, expected_rows, expected_cols):
        return grid

    return None


def _valid_grid(grid, expected_rows=None, expected_cols=None) -> bool:
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


# ─────────────────────────────────────────────
# Heurística de fallback inteligente
# ─────────────────────────────────────────────
def heuristic_fallback(task: dict, test_input: list) -> list:
    """
    Prueba transformaciones simples sobre el test input y devuelve la que
    mejor coincide con los ejemplos de train (o el primer output de train).
    Transformaciones: identity, rot90/180/270, flip horizontal, flip vertical.
    """
    arr = np.array(test_input)
    candidates = [
        arr,
        np.rot90(arr, 1),
        np.rot90(arr, 2),
        np.rot90(arr, 3),
        np.fliplr(arr),
        np.flipud(arr),
    ]

    train = task.get('train', [])
    if not train:
        return arr.tolist()

    # Si todos los train outputs tienen la misma forma, usamos esa como guía
    out_shapes = set(
        (len(ex['output']), len(ex['output'][0]))
        for ex in train if ex.get('output')
    )
    expected_shape = list(out_shapes)[0] if len(out_shapes) == 1 else None

    for c in candidates:
        if expected_shape and c.shape == expected_shape:
            return c.tolist()

    # Fallback final: primer output de train
    return train[0]['output']


# ─────────────────────────────────────────────
# Construcción de prompts
# ─────────────────────────────────────────────
def build_prompt(task: dict, test_input: list, compact: bool = False) -> str:
    """
    Prompt optimizado para ARC. Usa formato compacto si compact=True para
    ahorrar tokens con tareas grandes.
    """
    fmt = grid_to_compact if compact else grid_to_str

    system = (
        "You are an expert at solving ARC-AGI (Abstraction and Reasoning Corpus) tasks.\n"
        "Study the input→output pattern from ALL training examples, then predict the output for the test input.\n"
        "Rules:\n"
        "- Identify the transformation rule (color mapping, rotation, symmetry, counting, etc.).\n"
        "- Apply EXACTLY the same rule to the test input.\n"
        "- Output ONLY the final grid as a Python list of lists of integers.\n"
        "- No explanation after the grid. No markdown. Just the raw list.\n"
    )

    examples_block = ""
    for i, ex in enumerate(task.get('train', []), 1):
        examples_block += (
            f"Example {i}:\n"
            f"Input:\n{fmt(ex['input'])}\n"
            f"Output:\n{fmt(ex['output'])}\n\n"
        )

    test_block = (
        f"Test Input:\n{fmt(test_input)}\n\n"
        "Step-by-step reasoning:\n"
        "1. What is the transformation rule?\n"
        "2. Apply the rule to the test input.\n\n"
        "Output grid (list of lists of ints, NOTHING else after):\n"
    )

    full_prompt = system + "\n" + examples_block + test_block

    # Si el prompt es muy largo, usa formato compacto
    if not compact and len(full_prompt) > 8000:
        return build_prompt(task, test_input, compact=True)

    return full_prompt


# ─────────────────────────────────────────────
# ARCSolver principal
# ─────────────────────────────────────────────
class ARCSolver:
    def __init__(self, use_vllm: bool = False, vllm_config: dict = None):
        self.use_vllm = use_vllm
        self.llm = None

        if self.use_vllm and vllm_config:
            try:
                print(f"[ARCSolver] Cargando vLLM: {vllm_config['model']}")
                self.llm = LLM(
                    model=vllm_config['model'],
                    dtype=vllm_config['dtype'],
                    gpu_memory_utilization=vllm_config.get('gpu_memory_utilization', 0.85),
                    max_model_len=vllm_config.get('max_model_len', 12000),
                    trust_remote_code=True,
                )
                print("[ARCSolver] vLLM listo.")
            except Exception as e:
                print(f"[ARCSolver] ERROR cargando vLLM: {e}. Modo fallback activado.")
                self.use_vllm = False

    # ── Inferencia con vLLM ──────────────────
    def _generate(self, prompts: list, temperature: float = 0.3, n: int = 1) -> list[list[str]]:
        """
        Genera outputs para una lista de prompts.
        Devuelve lista de listas: [ [out1, out2, ...] por prompt ].
        """
        sp = SamplingParams(
            temperature=temperature,
            top_p=0.95,
            max_tokens=512,
            n=n,
            stop=["Input:", "Example", "Test Input:"],   # Evita que continúe generando ejemplos
        )
        results = self.llm.generate(prompts, sp)
        return [[o.text.strip() for o in r.outputs] for r in results]

    # ── Resolver un test case ────────────────
    def _solve_one(self, task: dict, test_input: list) -> list:
        prompt = build_prompt(task, test_input)
        exp_rows = len(test_input)
        exp_cols = len(test_input[0]) if test_input else 0

        # Intenta con voting de NUM_VOTES muestras
        for attempt, temp in enumerate(TEMPERATURES[:MAX_RETRIES + 1]):
            try:
                n = NUM_VOTES if attempt == 0 else 1
                outputs_batch = self._generate([prompt], temperature=temp, n=n)
                texts = outputs_batch[0]

                candidates = []
                for text in texts:
                    grid = parse_grid_from_text(text, exp_rows, exp_cols)
                    if grid:
                        candidates.append(grid)

                if candidates:
                    print(f"  [✓] Intento {attempt+1} OK – {len(candidates)}/{n} grids parseados (temp={temp})")
                    return majority_vote(candidates)

                print(f"  [✗] Intento {attempt+1} sin grids válidos (temp={temp}). Reintentando...")

            except Exception as e:
                print(f"  [!] Error en inferencia (intento {attempt+1}): {e}")

        # Fallback heurístico
        print("  [~] Usando fallback heurístico.")
        return heuristic_fallback(task, test_input)

    # ── Método principal ─────────────────────
    def solve(self, task: dict) -> list:
        if self.use_vllm and self.llm:
            predictions = []
            for idx, test_case in enumerate(task.get('test', [])):
                print(f"[Solving test {idx+1}/{len(task['test'])}]")
                pred = self._solve_one(task, test_case['input'])
                predictions.append(pred)
            return predictions

        # Modo fallback puro
        print("[ARCSolver] Modo fallback (sin vLLM).")
        return [
            heuristic_fallback(task, tc['input'])
            for tc in task.get('test', [])
        ]


# ─────────────────────────────────────────────
# Evaluación local (opcional)
# ─────────────────────────────────────────────
def evaluate(task: dict, predictions: list) -> dict:
    """Calcula accuracy si el task tiene soluciones en 'test'."""
    correct = 0
    total = len(predictions)
    for pred, test_case in zip(predictions, task.get('test', [])):
        if 'output' in test_case and grids_equal(pred, test_case['output']):
            correct += 1
    return {"correct": correct, "total": total, "accuracy": correct / total if total else 0}


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python arc_main.py task.json [--eval]")
        sys.exit(0)

    with open(sys.argv[1], 'r') as f:
        task = json.load(f)

    solver = ARCSolver(use_vllm=True, vllm_config=DEFAULT_VLLM_CONFIG)
    result = solver.solve(task)
    print("\n=== Predicciones ===")
    print(json.dumps(result, indent=2))

    if "--eval" in sys.argv:
        stats = evaluate(task, result)
        print(f"\n=== Evaluación: {stats['correct']}/{stats['total']} correctas ({stats['accuracy']*100:.1f}%) ===")
