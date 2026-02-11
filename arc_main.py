import json
import sys
import numpy as np
from vllm import LLM, SamplingParams

class ARCSolver:
    def __init__(self, use_vllm=False, vllm_config=None):
        self.use_vllm = use_vllm
        self.llm = None
        self.sampling_params = None
        if self.use_vllm and vllm_config:
            try:
                print("Cargando vLLM con config:", vllm_config)
                self.llm = LLM(
                    model=vllm_config['model'],
                    dtype=vllm_config['dtype'],
                    gpu_memory_utilization=vllm_config['gpu_memory_utilization'],
                    max_model_len=vllm_config['max_model_len']
                )
                self.sampling_params = SamplingParams(
                    temperature=0.7,
                    top_p=0.95,
                    max_tokens=1024
                )
                print("vLLM cargado exitosamente.")
            except Exception as e:
                print(f"Error cargando vLLM: {e}. Cayendo a modo dummy.")
                self.use_vllm = False

    def grid_to_string(self, grid):
        """Convierte un grid (list of lists) a string legible para el prompt."""
        return '\n'.join(' '.join(map(str, row)) for row in grid)

    def solve(self, task):
        """task: dict con 'train' (lista de dicts 'input'/'output') y 'test' (lista de dicts 'input')."""
        if self.use_vllm and self.llm:
            # Construye prompt para LLM
            prompt = "Eres un experto en razonamiento abstracto ARC-AGI. Analiza los ejemplos de train y predice el output para test.\n\n"
            for i, example in enumerate(task.get('train', [])):
                prompt += f"Ejemplo {i+1}:\nInput:\n{self.grid_to_string(example['input'])}\nOutput:\n{self.grid_to_string(example['output'])}\n\n"

            predictions = []
            for test_case in task.get('test', []):
                prompt_test = prompt + f"Test Input:\n{self.grid_to_string(test_case['input'])}\n"
                prompt_test += "Razonamiento step-by-step: Describe el pattern (colores, formas, transformaciones).\nLuego, genera el output grid como list of lists de ints (ej. [[1,0,1],[0,1,0],[1,0,1]]). Solo el grid final."

                # Inferencia con vLLM
                outputs = self.llm.generate([prompt_test], self.sampling_params)
                generated_text = outputs[0].outputs[0].text.strip()
                try:
                    # Extrae el grid del output (busca la parte que parece list of lists)
                    start = generated_text.rfind('[[')
                    end = generated_text.rfind(']]') + 2
                    grid_str = generated_text[start:end]
                    grid = json.loads(grid_str) if grid_str else np.zeros((3, 3), dtype=int).tolist()
                    predictions.append(grid)
                except Exception as e:
                    print(f"Error parseando output LLM: {e}. Usando fallback.")
                    predictions.append(np.zeros((len(test_case['input']), len(test_case['input'][0])), dtype=int).tolist())

            return predictions  # Lista de grids para cada test

        else:
            # Fallback dummy si no vLLM
            print("Usando modo dummy (sin vLLM).")
            predictions = []
            for test_case in task.get('test', []):
                # Copia primer train output o fallback
                if 'train' in task and task['train']:
                    predictions.append(task['train'][0]['output'])
                else:
                    predictions.append(np.zeros((3, 3), dtype=int).tolist())
            return predictions

if __name__ == "__main__":
    # Testing local: python arc_main.py task.json
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            task = json.load(f)
        vllm_config = {  # Simula tu .env config
            "model": "unsloth/Meta-Llama-3.1-8B-Instruct",
            "dtype": "half",
            "gpu_memory_utilization": 0.8,
            "max_model_len": 12000
        }
        solver = ARCSolver(use_vllm=True, vllm_config=vllm_config)
        result = solver.solve(task)
        print(json.dumps(result, indent=2))
    else:
        print("ARC Solver LLM listo. Uso: python arc_main.py task.json")
