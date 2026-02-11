import numpy as np
import json
import sys

class ARCSolver:
    def __init__(self, use_vllm=False):
        self.use_vllm = use_vllm
        print(f"ARC Solver dummy cargado (use_vllm={use_vllm})")

    def solve(self, task):
        """
        task: dict con 'train' (lista de ejemplos) y 'test' (lista de tests)
        Devuelve: lista de grids posibles (cada grid es list of lists de int)
        """
        # Dummy: copia el output del primer ejemplo de train
        if 'train' in task and task['train']:
            example_output = task['train'][0]['output']
            return [example_output]  # una predicción
        # Fallback: grid negro 3x3
        return [np.zeros((3, 3), dtype=int).tolist()]

# Wrapper para testing local (opcional)
if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            task = json.load(f)
        solver = ARCSolver(use_vllm=True)  # simula tu config
        result = solver.solve(task)
        print(json.dumps(result, indent=2))
    else:
        print("ARC Solver dummy listo. Uso: python arc_main.py task.json")
