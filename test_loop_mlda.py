import re
import time
import argparse
import subprocess
import os
import os

dir_path = os.path.dirname(os.path.realpath(__file__))


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Test")
    parser.add_argument("--start", "-s", type=int, default=0)
    parser.add_argument("--to", "-t", type=int, default=0)
    parser.add_argument("--eps", "-e", type=int, default=1)
    parser.add_argument("--out", "-o", type=str, required=True)
    args = parser.parse_args()

    filename = args.out

    print(f"==== Run {args.eps} times starting from {args.start} to {args.to} ======")
    print(f"Using algorithm MLDA")
    print(f"Record into CSV file: {filename}.csv")

    if input("Check before proceed. Press Y to continue. ").lower() != "y":
        exit()

    run_file_path = os.path.join(dir_path, "run_rviz_auto.py")

    for i in range(args.eps):
        for j in range(args.start, args.to + 1):
            process = subprocess.run(
                [
                    "python",
                    run_file_path,
                    "--world_idx",
                    f"{j}",
                    "--out",
                    f"{filename}.csv",
                ]
            )

    with open("test_log.txt", "a") as f:
        f.write(f"=============================\n")
        f.write(f"Run {args.eps} times in WORLDS {args.start} to {args.to}\n")
        f.write(f"With RVIZ {args.rviz}, using mlda 4th May\n")
        f.write(f"Record into CSV file: {args.out}.csv\n")
        duration = round(time.time() - start_time, 2)
        f.write(f"{(args.to - args.start +1)*args.eps} runs takes {duration} s\n")
        f.write(
            f"Or {round(duration/60,2)} minutes or {round(duration/3600,4)} hours\n"
        )
        f.write("\n\n")
