import wandb
import os
import subprocess
import selectors
import difflib
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
    def __init__(self, X, metadata, model, paper="full") -> None:
        self.X = X
        self.workspace = X["path"]
        self.mode = metadata["mode"]
        self.model = model
        self.paper = paper

    def run(self):
        if self.mode == "PC+refsol":
            return self.run_PC_refsol()
        elif self.mode == "FC+refsol":
            return self.run_FC_refsol()
        elif self.mode == "FC":
            return self.run_FC()
        elif self.mode == "PC":
            return self.run_PC()
        return 0

    def retrieve_information(self):
        func_removed = self.X["funcs_to_block"][0]
        file_name = func_removed["file"]
        func_name = func_removed["name"]
        line_start, header_line, line_end = func_removed["line_start"], func_removed["header_line"], func_removed["line_end"]

        with open(os.path.join(self.workspace, file_name), 'r') as file:
            contents = file.readlines()
        function_docstring = "\n".join(contents[header_line:line_start])

        with open(os.path.join(self.workspace, "paper.txt"), "r") as file:
            contents = file.readlines()
        contents = "\n".join(contents)

        prompt = f"You are tasked to write a Python function, but the details on how to write it is in this long text. Extract only the necessary context from the text. Python function: {function_docstring}. Text: {contents}"
        information = call_llm([{"role": "user", "content": prompt}], tools=None, model=self.model).content
        print(f"Relevant information: {information}")
        return information

    def write_missing_function(self):
        func_removed = self.X["funcs_to_block"][0]
        file_name = func_removed["file"]

        if self.paper == "no":
            edit_instruction = "Fill in the NotImplemented function"
        elif self.paper == "full":
            edit_instruction = f"Fill in the NotImplemented function according to these details: {self.retrieve_information()}"
        else:
            edit_instruction = f"Fill in the NotImplemented function according to these details: {func_removed['relevant_paper']}"


        content = open(os.path.join(self.workspace, file_name)).read()
        prompt = f"""Given this script:
        ```
        {content}
        ```
        Edit the script by following the instruction:
        {edit_instruction}
        Provide the full code after the edit, making no other changes. Provide only the code in markdown format. E.g. ```python
        """

        completion = call_llm([{"role": "system", "content": prompt}], tools=None, model=self.model).content

        new_content = completion.strip()
        if "```python" in new_content:
            new_content = new_content.split("```python")[1].split("```")[0].strip()
        if "```bash" in new_content:
            new_content = new_content.split("```bash")[1].split("```")[0].strip()
        if "```" in new_content:
            new_content = new_content.split("```")[1]

        with open(os.path.join(self.workspace, file_name), "w") as file:
            file.write(new_content)

        diff = list(difflib.unified_diff(content.splitlines(keepends=True), new_content.splitlines(keepends=True)))
        diff = "".join(diff)

        return diff 

    def extract_answer(self, output):
        prompt = f"Given this output of a script, retrieve the outputs of these experiments: {self.X['experiment_description']}. \nScript output: {output} \nPlease return the answer as a JSON string with keys \"Experiment 1\", \"Experiment 2\", ... and the values are the result of each experiment in JSON format." 
        answer = call_llm([{"role": "user", "content": prompt}], tools=None, model=self.model).content
        if "```json" in answer:
            answer = answer.split("```json")[1].split("```")[0].strip()
        return answer 

    def run_refsol(self):
        observation = command_line("export MKL_SERVICE_FORCE_INTEL=1 && bash -u refsol.sh", self.workspace)
        return self.extract_answer(observation)
        
    def generate_command_sequence(self):
        observation = command_line("tree -P '*.py' .", self.workspace)
        prompt = f"Given the current repository structure and the experiment we want to run, return the path to the Python script that needs to be run. Experiment: {self.X['experiment_description']}\n\nRepository structure: {observation}"
        answer = call_llm([{"role": "user", "content": prompt}], tools=None, model=self.model).content
        file_path = answer.split("```")[1].split("```")[0].strip()
        print(file_path)
        try:
            with open(os.path.join(self.workspace, file_path), "r") as f:
                contents = f.read()
            print(contents)
            prompt = f"Given the contents of a Python file we want to run and the experiment details, return the python command to run with the right arguments. Give your answer in this format ```bash\ncommand\n```. File path: {answer}\n\nFile contents: {contents}\n\nExperiment: {self.X['experiment_description']}"
            answer = call_llm([{"role": "user", "content": prompt}], tools=None, model=self.model).content
            print(answer) 
            if "```" in answer:
                answer = answer.split("```bash")[1].split("```")[0].strip()
            obs = command_line(f"export MKL_SERVICE_FORCE_INTEL=1 && {answer}", workspace=self.workspace)
            return self.extract_answer(obs)
        except Exception as e:
            return f"Got exception: {e}"

    def run_FC(self):
        return self.generate_command_sequence()
    
    def run_FC_refsol(self):
        return self.run_refsol()

    def run_PC(self):
        diff = self.write_missing_function()
        print(diff)
        answer = self.generate_command_sequence()
        return answer

    def run_PC_refsol(self):
        """ Partial Code + refsol """
        diff = self.write_missing_function()
        answer = self.run_refsol()
        return answer
        
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
