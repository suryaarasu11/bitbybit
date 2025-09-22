import numpy as np
import csv
from joblib import Parallel, delayed
import time
import sys

import chipwhisperer as cw

# ----------------------------- 
# 1. LOAD TRACES 
# -----------------------------
TRACE_CSV = "data.csv"

def load_traces(trace_file):
    traces = []
    ciphertexts = []
    try:
        with open(trace_file, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  # skip header
            for row in reader:
                c = int(row[0])
                ct = np.array([float(x) for x in row[2:]])
                traces.append(ct)
                ciphertexts.append(c)
    except FileNotFoundError:
        print(f"Error: File '{trace_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
    return np.array(traces), np.array(ciphertexts)

# -----------------------------
# 2. Hamming Weight Function
# -----------------------------
def hamming_weight(x):
    return bin(x).count("1")

# -----------------------------
# 3. CPA FUNCTION FOR ONE WINDOW AND BIT
# -----------------------------
def cpa_bit_window(bit_guess, recovered_bits_so_far, ciphertexts, traces, window_start, window_len):
    try:
        window_end = window_start + window_len
        predicted_hw = []
        for c in ciphertexts:
            val = 1
            bits_so_far = recovered_bits_so_far + [bit_guess]
            for b in bits_so_far:
                val = (val * val) % 64507
                if b == 1:
                    val = (val * c) % 64507
            predicted_hw.append(hamming_weight(val))
        predicted_hw = np.array(predicted_hw)
        trace_window = traces[:, window_start:window_end].mean(axis=1)
        if np.std(predicted_hw) == 0 or np.std(trace_window) == 0:
            corr = 0
        else:
            corr = np.corrcoef(predicted_hw, trace_window)[0,1]
        return np.abs(corr), bit_guess, window_start
    except Exception as e:
        print(f"Error in CPA calculation: {e}")
        return 0, bit_guess, window_start

# -----------------------------
# 4. AUTOMATIC BIT RECOVERY
# -----------------------------
def recover_key(traces_tuple):
    traces, ciphertexts = traces_tuple
    recovered_bits = []
    window_len = 20
    scan_step = 5
    num_traces, num_samples = traces.shape
    for i in range(15):  # 15-bit key
        results = Parallel(n_jobs=-1)(
            delayed(cpa_bit_window)(bit_guess, recovered_bits, ciphertexts, traces, w, window_len)
            for bit_guess in [0,1]
            for w in range(0, num_samples-window_len, scan_step)
        )
        max_corr, best_bit, best_window = max(results, key=lambda x: x[0])
        recovered_bits.append(best_bit)
        print(f"Recovered bit {i+1}: {best_bit} (corr={max_corr:.4f}, window_start={best_window})")
    d_bin = "".join(str(b) for b in recovered_bits)
    d_int = int(d_bin, 2)
    print(f"\nRecovered key bits: {recovered_bits}")
    print(f"Recovered private key d = {d_int}")
    return d_int

# -----------------------------
# 5. VERIFY KEY ON DEVICE
# -----------------------------
def verify_key(recovered_key):
    scope = cw.scope()
    scope.default_setup()
    target = cw.target(scope)
    prog = cw.programmers.STM32FProgrammer
    cw.program_target(scope, prog, "simpleserial_rsa-CW308_STM32F3.hex")
    time.sleep(1)
    ct_bytes = recovered_key.to_bytes(2, 'big')
    scope.arm()
    target.simpleserial_write('p', ct_bytes)
    ret = scope.capture()
    if ret:
        print("Verification capture timeout")
        return False
    resp = target.simpleserial_read('r', 2)
    if resp:
        plaintext = int.from_bytes(resp, 'big')
        print(f"Verification plaintext: {plaintext}")
        return plaintext == 6267
    return False

# -----------------------------
# 6. MAIN SCRIPT
# -----------------------------
def main():
    start_time = time.time()
    print("Loading traces...")
    traces_tuple = load_traces(TRACE_CSV)
    print(f"Loaded {traces_tuple[0].shape[0]} traces with {traces_tuple[0].shape[1]} samples each.")

    print("Recovering key bits...")
    key = recover_key(traces_tuple)
    print(f"Recovered key (decimal): {key}")
    print(f"Recovered key (bin): {bin(key)}")

    print("Verifying key...")
    if verify_key(key):
        print("Key verification successful! The recovered key is correct.")
    else:
        print("Key verification failed. Try refining your analysis.")

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\nTotal execution time: {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()

