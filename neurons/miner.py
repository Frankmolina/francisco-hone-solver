import bittensor as bt
import time
from arc_main import ARCSolver

# ─── Configuración ───────────────────────────────────────
WALLET_NAME = "mi_wallet"       # ← cambia esto por tu wallet real
HOTKEY_NAME = "default"         # ← cambia esto por tu hotkey real
NETUID      = 5
PORT        = 8091

VLLM_CONFIG = {
    "model": "unsloth/Meta-Llama-3.1-8B-Instruct",
    "dtype": "half",
    "gpu_memory_utilization": 0.85,
    "max_model_len": 12000,
}

# ─── Setup ───────────────────────────────────────────────
bt.logging.set_debug(True)

wallet    = bt.wallet(name=WALLET_NAME, hotkey=HOTKEY_NAME)
subtensor = bt.subtensor(network="finney")
metagraph = subtensor.metagraph(netuid=NETUID)

bt.logging.info(f"Wallet: {wallet}")
bt.logging.info(f"UID en SN5: {metagraph.hotkeys.index(wallet.hotkey.ss58_address) if wallet.hotkey.ss58_address in metagraph.hotkeys else 'NO REGISTRADO'}")

# ─── Cargar solver (solo UNA vez, fuera del handler) ─────
solver = ARCSolver(use_vllm=False, vllm_config=VLLM_CONFIG)
# ↑ use_vllm=False porque vLLM no funciona en Mac M2
# Para usar el modelo necesitas cambiar ARCSolver a transformers+MPS

# ─── Axon y handler ──────────────────────────────────────
axon = bt.axon(wallet=wallet, port=PORT)

def solve_arc_task(synapse):
    try:
        bt.logging.info("Tarea recibida del validador")
        task = synapse.task if hasattr(synapse, 'task') else synapse.dict()
        result = solver.solve(task)
        synapse.predictions = result
        bt.logging.success(f"Predicciones enviadas: {len(result)} outputs")
    except Exception as e:
        bt.logging.error(f"Error procesando tarea: {e}")
    return synapse

axon.attach(forward_fn=solve_arc_task)

# ─── Servir y mantener vivo ───────────────────────────────
axon.serve(netuid=NETUID, subtensor=subtensor)
axon.start()

bt.logging.success(f"Miner activo en puerto {PORT} — SN5")

# Sin este loop el proceso muere inmediatamente
while True:
    metagraph.sync(subtensor=subtensor)
    bt.logging.info(f"Bloque actual: {subtensor.block}")
    time.sleep(60)
