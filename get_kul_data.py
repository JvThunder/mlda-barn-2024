import argparse
import subprocess

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-s', type=int, default=0)
parser.add_argument('-t', type=int, default=4)
args = parser.parse_args()

start = args.s
end = args.t

print(f"==== Run starting from {start} to {end} ======")

for j in range(start, end + 1):
    print(f"==== Running world {j} ====")
    for attempt in range(5):  # Retry up to 5 times
        result = subprocess.run(["python", "run_rviz_kul.py", "--world_idx", str(j)])
        if result.returncode != 400:  # Break the loop if the return code is not 400
            break
        print(f"Attempt {attempt + 1} failed with exit code 400, retrying...")
    print(f"==== Done with world {j} ====")