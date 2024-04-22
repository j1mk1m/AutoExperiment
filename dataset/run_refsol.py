import os
import subprocess
import time

def run_refsol(X):
    start = time.time()
    refsol = os.path.join(X["path"], "bash.sh")
    subprocess.run(["bash", refsol])
    end = time.time()
    print(f"Run time: {end - start} seconds\n\n")
