import argparse
import subprocess
import time

SINGLE_ENV_TRAIN_SIZE = 3
for j in range(0, 300, 1):
    for _ in range(SINGLE_ENV_TRAIN_SIZE):
        print(f"==== Running world {j} ====")
        success = False
        while not success:
            try:
                result = subprocess.run(["python", "run_rviz_kul.py", "--world_idx", str(j)],
                                        timeout=100)
                if result.returncode == 200:
                    print("==== Success ====")
                    success = True
                else:
                    print("Fail... retrying")
                time.sleep(5)
            except subprocess.TimeoutExpired:
                print("Timeout... retrying")
                # Kill the process
                subprocess.run(["pkill", "-f", "rviz"])
                time.sleep(5)
        