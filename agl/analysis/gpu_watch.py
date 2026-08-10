"""Passive GPU power/thermal watchdog. Zero GPU load (read-only queries).

Writes one line per second: <unix_ts> <power_w> <temp_c> <util%> <mem_mib>.
On a hard system hang the log simply stops; the LAST line = the state at the
freeze moment, which distinguishes a power-budget brownout (high power at
freeze) from a silicon lockup (freeze at arbitrary power).
"""
import subprocess
import time

LOG = "/home/zhaoguodong/work/code/AutonomousGapLearning/runs/gpu_watch.log"
Q = ["nvidia-smi", "--query-gpu=power.draw,temperature.gpu,utilization.gpu,memory.used",
     "--format=csv,noheader,nounits"]


def main():
    while True:
        line = ""
        try:
            out = subprocess.check_output(Q, stderr=subprocess.DEVNULL).decode().strip()
            line = f"{int(time.time())} {out.replace(',', ' ')}"
        except Exception as e:
            line = f"{int(time.time())} QUERY_ERR {type(e).__name__}"
        with open(LOG, "a") as f:
            f.write(line + "\n")
        time.sleep(1)


if __name__ == "__main__":
    main()
