import bittensor as bt
from arc_main import ARCSolver

wallet = bt.wallet(name="mi_wallet", hotkey="mi_hotkey")
subtensor = bt.subtensor(network="finney")
axon = bt.axon(wallet=wallet, port=8091)

# Registrar el handler que recibe tareas del validador
@axon.attach
def solve_arc_task(synapse):
    solver = ARCSolver(use_vllm=True, ...)
    result = solver.solve(synapse.task)
    synapse.predictions = result
    return synapse

axon.serve(netuid=5, subtensor=subtensor)
axon.start()
