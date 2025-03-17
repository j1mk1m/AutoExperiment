import os
import subprocess
import time
import selectors


def run_refsol(X):
    start = time.time()
    command_line(f"export MKL_SERVICE_FORCE_INTEL=1 && bash refsol.sh", X["path"])
    end = time.time()
    print(f"Run time: {end - start} seconds\n\n")

def command_line(command, workspace):
    command = command.strip()
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True, cwd=workspace)

        stdout_lines = []
        stderr_lines = []
        lines = []

        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        selector.register(process.stderr, selectors.EVENT_READ)

        while process.poll() is None and selector.get_map():
            events = selector.select(timeout=1)

            for key, _ in events:
                line = key.fileobj.readline()
                if key.fileobj == process.stdout:
                    print("STDOUT:", line, end =" ")
                    stdout_lines.append(line)
                    lines.append(line)
                else:
                    print("STDERR:", line, end =" ")
                    stderr_lines.append(line)
                    lines.append(line)

        for line in process.stdout:
            line = line
            print("STDOUT:", line, end =" ")
            stdout_lines.append(line)
            lines.append(line)
        for line in process.stderr:
            line = line
            print("STDERR:", line, end =" ")
            stderr_lines.append(line)
            lines.append(line)

        return_code = process.returncode

        if return_code != 0:
            observation = "".join(lines)
        else:
            observation = "".join(lines)
        if observation == "" and return_code == 0:
            # printed to stderr only
            observation = "".join(lines)
        
        return observation
    except Exception as e:
        return f"Something went wrong in executing {command}: {e}."
