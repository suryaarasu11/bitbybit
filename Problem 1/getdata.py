import chipwhisperer as cw
import random
import csv
import time
import logging
import os

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # --- 1. CONNECT TO HARDWARE ---
    try:
        logging.info("Connecting to ChipWhisperer-Lite and target device...")
        scope = cw.scope()
        scope.default_setup()
        target = cw.target(scope)
    except Exception as e:
        logging.error(f"Failed to connect to ChipWhisperer or target device: {e}")
        return

    # --- 2. PROGRAM THE TARGET ---
    prog = cw.programmers.STM32FProgrammer
    firmware_file = "simpleserial_rsa-CW308_STM32F3.hex"
    logging.info(f"Programming target with firmware: {firmware_file} ...")
    try:
        cw.program_target(scope, prog, firmware_file)
        logging.info("Programming done.")
    except Exception as e:
        logging.error(f"Programming failed: {e}")
        return

    time.sleep(1)  # Allow MCU reboot

    # --- 3. CONFIGURE CAPTURE SETTINGS ---
    scope.clock.adc_src = "clkgen_x1"  # Confirm this matches your firmware/setup
    scope.adc.samples = 500
    scope.adc.timeout = 2000

    # --- 4. PREPARE FOR CAPTURE ---
    RSA_N = 64507
    NUM_TRACES = 500
    OUT_CSV = "data1.csv"

    output_dir = os.path.dirname(os.path.abspath(OUT_CSV))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")

    if os.path.exists(OUT_CSV):
        logging.info(f"CSV file {OUT_CSV} exists and will be overwritten")
    
    random.seed(0xCAFEBABE)
    rows = []

    # --- 5. CAPTURE LOOP ---
    logging.info(f"Starting trace capture for {NUM_TRACES} random ciphertexts...")
    for i in range(NUM_TRACES):
        c_int = random.randint(0, RSA_N - 1)
        ct_bytes = c_int.to_bytes(2, 'big')

        try:
            scope.arm()
            target.simpleserial_write('p', ct_bytes)

            ret = scope.capture()
            if ret:
                # Timeout or capture failure
                logging.warning(f"Capture timed out or failed for trace {i + 1}")
                continue

            resp = target.simpleserial_read('r', 2)
            if resp:
                plaintext = int.from_bytes(resp, 'big')
            else:
                plaintext = None
                logging.warning(f"No plaintext response for trace {i + 1}")

            trace = scope.get_last_trace()
            if trace is None or len(trace) == 0:
                logging.warning(f"Invalid or empty trace at {i + 1}")
                continue

            row_data = [c_int, plaintext] + list(trace)
            rows.append(row_data)

            time.sleep(0.01)  # Stability between captures

            if (i + 1) % 100 == 0:
                logging.info(f"Captured {i+1}/{NUM_TRACES} traces...")

        except Exception as e:
            logging.warning(f"Error during trace {i+1}: {e}")
            continue

    # --- 6. SAVE CAPTURED DATA TO CSV ---
    logging.info(f"Saving {len(rows)} traces to {OUT_CSV}...")
    try:
        with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            header = ['ciphertext', 'plaintext'] + [f'sample_{i}' for i in range(scope.adc.samples)]
            writer.writerow(header)
            writer.writerows(rows)
        logging.info(f"Successfully saved traces to {OUT_CSV} ({os.path.getsize(OUT_CSV):,} bytes)")
    except Exception as e:
        logging.error(f"Failed to save CSV: {e}")
        # Attempt backup save
        backup_csv = f"my_traces_backup_{int(time.time())}.csv"
        try:
            with open(backup_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                header = ['ciphertext', 'plaintext'] + [f'sample_{i}' for i in range(scope.adc.samples)]
                writer.writerow(header)
                writer.writerows(rows)
            logging.info(f"Saved backup traces to {backup_csv}")
        except Exception as backup_e:
            logging.error(f"Backup save failed: {backup_e}")

    # --- 7. BASIC STATISTICS ---
    if rows:
        ciphertexts = [row[0] for row in rows]
        plaintexts = [row[1] for row in rows if row[1] is not None]

        logging.info(f"Statistics:")
        logging.info(f"  Total traces captured: {len(rows)}")
        logging.info(f"  Successful plaintext responses: {len(plaintexts)}")
        logging.info(f"  Ciphertext range: {min(ciphertexts)} - {max(ciphertexts)}")
        if plaintexts:
            logging.info(f"  Plaintext range: {min(plaintexts)} - {max(plaintexts)}")

if __name__ == "__main__":
    main()

