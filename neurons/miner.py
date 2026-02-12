import bittensor as bt
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arc_main import ARCSolver
from typing import Optional

# ─── Configuración ───────────────────────────────────────
WALLET_NAME = "minador_cold"
HOTKEY_NAME = "minador_hot"
NETUID      = 5
PORT        = 8091
MY_UID      = 123

# ─── Synapse personalizado para ARC ──────────────────────
class ARCSynapse(bt.Synapse):
    task: Optional[dict] = None
    predictions: Optional[list] = None

# ─── Setup ───────────────────────────────────────────────
bt.logging.set_debug(True)

wallet    = bt.Wallet(name=WALLET_NAME, hotkey=HOTKEY_NAME)
subtensor = bt.Subtensor(network="finney")
metagraph = subtensor.metagraph(netuid=NETUID)

bt.logging.info(f"Wallet: {wallet}")
bt.logging.info(f"Hotkey: {wallet.hotkey.ss58_address}")

# ─── Cargar solver una sola vez ──────────────────────────
solver = ARCSolver(use_vllm=False)

# ─── Axon y handler ──────────────────────────────────────
axon = bt.Axon(wallet=wallet, port=PORT)

def solve_arc_task(synapse: ARCSynapse) -> ARCSynapse:
    try:
        bt.logging.info("Tarea recibida del validador")
        result = solver.solve(synapse.task or {})
        synapse.predictions = result
        bt.logging.success(f"Predicciones enviadas: {len(result)} outputs")
    except Exception as e:
        bt.logging.error(f"Error: {e}")
    return synapse

axon.attach(forward_fn=solve_arc_task)

# ─── Servir y mantener vivo ───────────────────────────────
axon.serve(netuid=NETUID, subtensor=subtensor)
axon.start()

bt.logging.success(f"✓ Miner activo — UID {MY_UID} — Puerto {PORT}")

while True:
    try:
        metagraph.sync(subtensor=subtensor)
        my_incentive = metagraph.I[MY_UID].item()
        my_trust     = metagraph.TS[MY_UID].item()
        bt.logging.info(
            f"Bloque: {subtensor.block} | "
            f"UID {MY_UID} | "
            f"Incentive: {my_incentive:.6f} | "
            f"Trust: {my_trust:.6f}"
        )
    except Exception as e:
        bt.logging.warning(f"Error en sync, reintentando en 60s: {e}")
    time.sleep(60)
