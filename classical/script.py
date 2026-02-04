from client import GameClient
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import json
from pathlib import Path
import numpy as np

def get_session_path(username):
    """Generates a filename based on the username."""
    # Replaces spaces/special chars to keep the filename clean
    safe_name = "".join(x for x in username if x.isalnum())
    return Path(f"session_{safe_name}.json")

def save_session(client):
    if client.api_token:
        # Dynamic filename based on client name
        session_file = get_session_path(client.player_id)
        with open(session_file, "w") as f:
            json.dump({
                "api_token": client.api_token, 
                "player_id": client.player_id, 
                "name": client.name
            }, f)
        print(f"Session saved to {session_file}.")

def load_session(player_id):
    session_file = get_session_path(player_id)
    if not session_file.exists():
        return None
    with open(session_file) as f:
        data = json.load(f)
    client = GameClient(api_token=data.get("api_token"))
    client.player_id = data.get("player_id")
    client.name = data.get("name")
    status = client.get_status()
    if status:
        print(f"Resumed: {client.player_id} | Score: {status.get('score', 0)} | Budget: {status.get('budget', 0)}")
        return client
    return None

# --- Main Execution ---

# 1. Ask for User Input first to check for an existing session
input_id = input("Enter your Player ID: ").strip()

client = load_session(input_id)
chosen_candidate = ""
if not client:
    print(f"No saved session for '{input_id}'. Register below.")
    
    # 2. Gather remaining info for registration
    input_name = input("Enter your Full Display Name: ").strip()
    
    client = GameClient()
    
    # Using the inputs provided by the user
    PLAYER_ID = input_id
    PLAYER_NAME = input_name
    
    result = client.register(PLAYER_ID, PLAYER_NAME, location="in_person")
    
    if result.get("ok"):
        print(f"Registered! Token: {client.api_token[:20]}...")
        candidates = result["data"].get("starting_candidates", [])
        print(f"\nStarting candidates ({len(candidates)}):")
        for c in candidates:
            print(f"  - {c['node_id']}: {c['utility_qubits']} qubits, +{c['bonus_bell_pairs']} bonus")
        chosen_candidate = candidates[0]['node_id']
        save_session(client)
    else:
        print(f"Failed: {result.get('error', {}).get('message')}")
else:
    print(f"Already registered as {client.player_id}")

status = client.get_status()


if status.get('starting_node'):
    start = status['starting_node']
    print(f"Starting node: {status['starting_node']}")
    print(f"Budget: {status['budget']} | Score: {status['score']}")
else:
    print("Select a starting node from the candidates shown above.")
    # Uncomment and modify:
    start = client.select_starting_node(chosen_candidate)
    start = start['data']['starting_node']
client.print_status()

from qiskit.circuit.classical import expr

def create_fake_circuit(N):
    qr = QuantumRegister(N * 2, 'q')  
    cr = ClassicalRegister(N * 2 - 1, 'c')  # Classical bits for measurements + flag
    qc = QuantumCircuit(qr, cr)
    qc.cx(qr[1], qr[0])
    qc.cx(qr[2], qr[3])
    qc.measure(qr[0], cbit=cr[0])
    qc.measure(qr[3], cbit=cr[1])
    return qc
    
def create_level_n_circuit(N):
    """Example distillation circuit template for 3 Bell pairs."""
    target_a = N - 1
    target_b = N
    qr = QuantumRegister(N * 2, 'q')  
    cr = ClassicalRegister(N * 2 - 1, 'c')  # Classical bits for measurements + flag
    qc = QuantumCircuit(qr, cr)
    
    # Qubit layout for N=3:
    #   q0, q5: Ancilla pair
    #   q1, q4: Ancilla pair
    #   q2, q3: Data pair
    for i in range(N):
        qc.rx(np.pi / 2, qr[i])
        qc.rx(-np.pi / 2, qr[2 * N - 1 - i])

    for i in range(N - 1):
        qc.cx(qr[target_a], qr[i])
        qc.cx(qr[target_b], qr[2 * N - 1 - i])
    
        qc.measure(qr[i], cr[i * 2 + 1])
        qc.measure(qr[2 * N - 1 - i], cr[i * 2 + 2])
    
        qc.barrier()

    for i in range(N - 1):
        qc.store(cr[i * 2 + 1], expr.bit_xor(cr[i * 2 + 1], cr[i * 2 + 2]))
        qc.store(cr[0], expr.bit_or(cr[0], cr[i * 2 + 1]))
        # qc.store(cr[0], expr.bit_not(cr[0]))

    return qc

import time
import requests

# Track edges locally to prevent infinite loops if the server cache is slow
processed_edges = set()

import sys
import subprocess
import json

cmd = [
    sys.executable, "classical.py",
    "--nodes", "/Users/kephasher/Documents/2026-IonQ/data1.csv",
    "--edges", "/Users/kephasher/Documents/2026-IonQ/data2.csv",
    "--starts", f"{start}",
    "--time-limit", "10",
    "--seed", "1234",
    "--out-json", "output",
    "--difficulty-map", "1:2,2:2,3:3,4:3,5:1000",
    "--budget", "40",
]

subprocess.run(cmd, check=True)
print("Calculations done")


with open("output", "r", encoding="utf-8") as f:
    data = json.load(f)
    final_budget = data['final_budget']
    total_utility = data['total_utility']
    total_bonus = data['total_bonus']
    total_edge_cost = data['total_edge_cost']
    num_nodes = data['num_nodes']
    num_edges  = data['num_edges']
    edges_in_order = data["edges_in_order"]


step = 0

while True:
    claimable_list = client.get_claimable_edges()
    # Filter out edges we already successfully claimed in this session
    targets = [e for e in claimable_list if tuple(e['edge_id']) not in processed_edges]
    if not targets:
        print("No more new claimable edges found.")
        break

    threshold = edges_in_order[step]["threshold"]
    edge_id = (edges_in_order[step]["from"], edges_in_order[step]["to"])

    success_on_edge = False
    client.print_status()
    print(f"\n>>> Targeting: {edge_id} (Threshold: {threshold:.3f})")

    if edge_id not in [tuple(target['edge_id']) for target in targets]:
        edge_id = (edges_in_order[step]["to"], edges_in_order[step]["from"])

    # print([tuple(target['edge_id']) for target in targets])
    assert edge_id in [tuple(target['edge_id']) for target in targets]

    for n in range(2, 5):
        if success_on_edge:
            break
            
        circuit = create_level_n_circuit(n)
        fakecircuit = create_fake_circuit(n)
        flag_bit = 0 
        
        # Internal loop to handle 500 errors without increasing N
        max_retries = 5
        for attempt in range(max_retries):
            try:
                print(f"  Attempting with {n} Bell pairs...")
                client.print_status()
                result = client.claim_edge(edge_id, circuit, flag_bit, n)
                # result = client.claim_edge(edge_id, fakecircuit, flag_bit, n)
                # result = client.claim_edge(edge_id, circuit, flag_bit, n)
                if result.get("ok"):
                    data = result["data"]
                    print(f"  Fidelity: {data.get('fidelity', 0):.4f} | Success: {data.get('success')}")
                    
                    if data.get('success'):
                        print(f"  [!] Successfully claimed {edge_id}!")
                        processed_edges.add(edge_id)
                        success_on_edge = True
                        break # Exit retry loop
                    else:
                        print(f"  [-] Fidelity too low for threshold. Increasing N...")
                        break # Break retry loop to try n + 1
                else:
                    error_msg = result.get("error", {}).get("message", "").lower()
                    if "budget" in error_msg:
                        print("!!! OUT OF BUDGET !!! Stopping execution.")
                        exit() # Stop the script entirely
                    print(f"  [-] API Error: {error_msg}")
                    break
                    
            except (requests.exceptions.RequestException, Exception) as e:
                print(f"  [!] Network/Server Error: {e}. Retrying same N in 5s...")
                time.sleep(5)
                if attempt == max_retries - 1:
                    print(f"  [!] Max retries reached for {n} pairs. Skipping.")

    if not success_on_edge:
        # If we couldn't get it, mark it as processed so we don't get stuck in a loop
        processed_edges.add(edge_id)


    step += 1