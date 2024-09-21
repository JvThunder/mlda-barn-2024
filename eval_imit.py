import argparse
import subprocess
import time

print(f"==== Run starting from 0 to 299 ======")

RETRIES = 3
for j in range(0, 300, 5):
    print(f"==== Running world {j} ====")
    for attempt in range(RETRIES):  # Retry up to 3 times
        result = subprocess.run(["python", "run_rviz_imit.py", "--world_idx", str(j)])
        if result.returncode == 200:  # Break the loop if the return code is 200
            print("==== Success ====")
            break
        print(f"Attempt {attempt + 1} failed, retrying...")
        time.sleep(1)

    print(f"==== Done with world {j} ====")