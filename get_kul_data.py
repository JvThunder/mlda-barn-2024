import argparse
import subprocess

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-s', type=int, default=158)
parser.add_argument('-e', type=int, default=158)
args = parser.parse_args()

start = args.s
end = args.e

print(f"==== Run starting from {start} to {end} ======")

RETRIES = 100
for j in range(start, end + 1):
    print(f"==== Running world {j} ====")
    for attempt in range(RETRIES):  # Retry up to 5 times
        result = subprocess.run(["python", "run_rviz_kul.py", "--world_idx", str(j)])
        if result.returncode == 200:  # Break the loop if the return code is 200
            break
        print(f"Attempt {attempt + 1} failed, retrying...")
    print(f"==== Done with world {j} ====")