import wandb
import os
import subprocess
import selectors
from agents.BasicPromptAgent.llm import *

PROMPT = """
Given the following code, fill out the missing function.

File: {file_name}
```
{contents}
```

Fill out contents of function {func_name} at line {line}.
"""

class CodeAgent:
    def __init__(self, X, model) -> None:
        self.X = X
        self.workspace = X["path"]
        self.mode = X["mode"]
        self.model = model

    def run(self):
        if self.mode == "PC+refsol":
            return self.run_PC_refsol()
        elif self.mode == "FC+refsol":
            return self.run_FC_refsol()
        return 0
    
    def run_FC_refsol(self):
        # Run refsol
        observation = command_line("bash -u refsol.sh", self.workspace)
        wandb.log({"observation": observation})
        return "0.0"

    def run_PC_refsol(self):
        """ Partial Code + refsol """
        func_removed = self.X["func_to_block"]
        description = self.X["experiment_description"]

        # extract code
        file_name = func_removed["script"]
        func_name = func_removed["name"]
        line_start, line_end = func_removed["line_start"], func_removed["line_end"]

        with open(os.path.join(self.workspace, "code", file_name), 'r') as file:
            contents = file.readlines()
        
        # Prompt LLM to complete code
        prompt = PROMPT.format(file_name=file_name, contents="".join(contents), func_name=func_name, line=line_start)
        response = call_llm([{"role": "user", "content": prompt}], None, self.model)

        func_content = response.content
        func_content = func_content.split("```python\n")[1].split("```")[0] 
        print(func_content)
        func_content = func_content.split("\n")

        n = 0
        while contents[line_start-1][n].isspace():
            n += 1

        m = 0
        while func_content[0][m].isspace():
            m += 1

        if n == m:
            middle = [line + "\n" for line in func_content[1:]]
        else:
            n = n - m  # shift
            middle = [" " * n + line + "\n" for line in func_content[1:]]
        
        # Write generated code into file
        new_contents = contents[:line_start] + middle + contents[line_end:]

        with open(os.path.join(self.workspace, "code", file_name), 'w') as write_file:
            write_file.writelines(new_contents)

        # Run refsol
        observation = command_line("bash -u refsol.sh", self.workspace)
        wandb.log({"observation": observation})
        return 0.0

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
