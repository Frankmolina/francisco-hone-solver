import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bittensor as bt
import time
from arc_main import ARCSolver
# ... resto del códigoimport bittensor as bt
import time
from arc_main import ARCSolver

# ─── Configuración ───────────────────────────────────────
WALLET_NAME = "minador_cold"
HOTKEY_NAME = "minador_hot"
NETUID      = 5
PORT        = 8091
MY_UID      = 123

# ─── Setup ───────────────────────────────────────────────
bt.logging.set_debug(True)

wallet    = bt.wallet(name=WALLET_NAME, hotkey=HOTKEY_NAME)
subtensor = bt.subtensor(network="finney")
metagraph = subtensor.metagraph(netuid=NETUID)

bt.logging.info(f"Wallet: {wallet}")
bt.logging.info(f"Hotkey: {wallet.hotkey.ss58_address}")

# ─── Cargar solver una sola vez ──────────────────────────
solver = ARCSolver(use_vllm=False)

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

bt.logging.success(f"✓ Miner activo — UID {MY_UID} — Puerto {PORT}")

while True:
    metagraph.sync(subtensor=subtensor)
    my_incentive = metagraph.I[MY_UID].item()
    my_trust     = metagraph.T[MY_UID].item()
    bt.logging.info(
        f"Bloque: {subtensor.block} | "
        f"UID {MY_UID} | "
        f"Incentive: {my_incentive:.6f} | "
        f"Trust: {my_trust:.6f}"
    )
    time.sleep(60)
