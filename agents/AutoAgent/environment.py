import os
import shutil
import subprocess
import selectors
import difflib
import sys
this_dir = os.path.dirname(__file__)
sys.path.append(this_dir)

from llm import call_llm
from prompts import edit_file_prompt 

class Environment:
    def __init__(self, X, metadata, model, retrieval, **kwargs) -> None:
        self.X = X
        self.retrieval = retrieval
        self.metadata = metadata
        self.combined_id = metadata["combined_id"]
        self.mode = metadata["mode"]
        self.source = X["path"]
        self.model = model
        self.setup_workspace(self.source)
        self.cur_dir = self.workspace_root
        self.action_to_function_mapper = {
            "final_answer": self.final_answer,
            "read_file": self.read_file,
            "write_file": self.write_file,
            "edit_file": self.edit_file,
            "understand_file": self.understand_file,
            "inspect_file_lines": self.inspect_file_lines,
            "list_files": self.list_files,
            "move": self.move,
            "change_directory": self.change_directory,
            "execute_python_script": self.execute_python_script,
            "execute_bash_script": self.execute_bash_script,
            "command_line": self.command_line,
            "edit_missing_function": self.edit_missing_function,
            "run_experiment": self.run_experiment
        }

        self.remove_code_context = "remove_code_context" in kwargs and kwargs["remove_code_context"]

    def setup_workspace(self, source):
        self.workspace_root = os.path.normpath(os.path.join(this_dir, "workspace", f"{self.mode}_{self.combined_id}"))
        if os.path.exists(self.workspace_root):
            shutil.rmtree(self.workspace_root)

        shutil.copytree(source, self.workspace_root, symlinks=True)

    def get_exp_description(self):
        with open(os.path.join(self.source, "experiment.txt"), 'r') as file: 
            return file.read()

    def execute(self, action, inputs):
        if action not in self.action_to_function_mapper:
            return f"Tool {action} not supported"
        try:
            observation = self.action_to_function_mapper[action](**inputs)
        except Exception as e:
            observation = f"Execution of {action} resulted in error {e}"
        return observation

    # Actions
    def final_answer(self, final_answer, **kwargs):
        return f"Final answer submitted: {final_answer}"

    # File Manipulation
    def read_file(self, file_name, **kwargs):
        try:
            observation = open(os.path.join(self.cur_dir, file_name)).read()
            return observation
        except:
            return f"Cannot find file {file_name}. Tip: use list_files to see contents of the current directory"

    def understand_file(self, file_name, things_to_look_for="General summary", **kwargs):
        lines = self.read_file(file_name).split("\n")
        if f"Cannot find file {file_name}" in lines[0]:
            return lines[0]
        # group lines to blocks so that each block has at most 10000 characters
        counter = 0
        blocks = []
        while counter < len(lines):
            block = []
            start_line_number = counter + 1
            while counter < len(lines) and len("\n".join(block)) + len(lines[counter]) < 10000:
                block.append(lines[counter])
                counter += 1
            if len(block) > 0:
                end_line_number = counter 
                blocks.append(("\n".join(block), start_line_number, end_line_number))
            else:
                end_line_number = start_line_number
                # probably a file of one/few very long line; split by 10000 characters
                for i in range(0, len(lines[counter]), 10000):
                    blocks.append((lines[counter][i:i+10000], start_line_number, end_line_number))
                counter += 1

        descriptions  = []
        for idx, (b, start_line_number, end_line_number) in enumerate(blocks):
            start_char_number = sum([len(b) for b in blocks[:idx]])
            end_char_number = start_line_number + len(b)
            prompt = f"""Given this (partial) file from line {start_line_number} character {start_char_number} to line {end_line_number} character {end_char_number}: 
        ``` 
        {b}
        ```
        Here is a detailed description on what to look for and what should be returned: {things_to_look_for}
        The description should reference crtical lines in the script relevant to what is being looked for. Only describe what is objectively confirmed by the file content. Do not include guessed numbers. If you cannot find the answer to certain parts of the request, you should say "In this segment, I cannot find ...".
        """
            messages = [{"role": "system", "content": prompt}]
            completion = call_llm(messages, None, self.model).content
            descriptions.append(completion)
        if len(descriptions) == 1:
            return descriptions[0]
        else:
            descriptions = "\n\n".join(["Segment {idx}: \n\n" + s for s in descriptions])
            prompt = f"""Given the relevant observations for each segments of a file, summarize to get a cohesive description of the entire file on what to look for and what should returned: {things_to_look_for}
                {descriptions}
                """
 
        messages = [{"role": "system", "content": prompt}]
        completion = call_llm(messages, None, self.model).content
        return completion

    def inspect_file_lines(self, file_name, start_line_number=0, end_line_number=None, **kwargs):
        lines = self.read_file(file_name).split("\n")
        if f"Cannot find file {file_name}" in lines[0]:
            return lines[0]
        return "\n".join(lines[start_line_number:(end_line_number if end_line_number else len(lines))])

    def write_file(self, file_name, content, **kwargs):
        with open(os.path.join(self.cur_dir, file_name), 'w') as file:
            file.write(content)
        return "File written successfully"

    def edit_file(self, file_name, edit_instruction, save_name=None, **kwargs):
        try:
            content = self.read_file(file_name)
        except:
            self.write_file(file_name, "")
            content = ""

        if self.remove_code_context:
            header_line = self.X["func_details"][0]["header_line"]
            end_line = self.X["func_details"][0]["line_end"]
            content = "\n".join(content.split("\n")[header_line-1:end_line])

        if self.retrieval == "oracle":
            edit_instruction += "\n" + self.X["funcs_to_block"][0]["relevant_paper"]

        prompt = f"""Given this script:
        ```
        {content}
        ```
        Edit the script by following the instruction:
        {edit_instruction}
        Provide the full code after the edit, making no other changes. Provide only the code in markdown format. E.g. ```python or ```bash
        """

        completion = call_llm([{"role": "system", "content": prompt}], None, self.model).content

        new_content = completion.strip()
        if "```python" in new_content:
            new_content = new_content.split("```python")[1].split("```")[0].strip()
        if "```bash" in new_content:
            new_content = new_content.split("```bash")[1].split("```")[0].strip()
        if "```" in new_content:
            new_content = new_content.split("```")[1]

        if save_name is None: save_name = file_name
        self.write_file(save_name, new_content)

        diff = list(difflib.unified_diff(content.splitlines(keepends=True), new_content.splitlines(keepends=True)))
        diff = "".join(diff)

        return f"The edited file is saved to {save_name}. Here is the diff, please check if the edit is correct and desirable:\n\n" + diff
    
    def edit_missing_function(self, function_number, instruction, **kwargs):
        func_removed = self.X["funcs_to_block"][function_number-1]
        description = self.X["experiment_description"]

        # extract code
        file_name = func_removed["file"]
        func_name = func_removed["name"]
        header_line, line_start, line_end = func_removed["header_line"], func_removed["line_start"], func_removed["line_end"]

        with open(os.path.join(self.workspace_root, file_name), 'r') as file:
            contents = file.readlines()
        
        # Prompt LLM to complete code
        prompt = edit_file_prompt.format(file_name=file_name, contents="".join(contents), func_name=func_name, line=line_start, instruction=instruction)
        response = call_llm([{"role": "user", "content": prompt}], None, self.model)

        func_content = response.content
        if "```python" in func_content:
            func_content = func_content.split("```python\n")[1].split("```")[0] 
        elif "```" in func_content:
            func_content = func_content.split("```")[1]
        func_content = func_content.split("\n")

        n = 0
        while contents[header_line-1][n].isspace():
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
        new_contents = contents[:line_start-1] + middle + contents[line_end:]

        func_removed["line_end"] = line_start + len(middle)

        with open(os.path.join(self.workspace_root, file_name), 'w') as write_file:
            write_file.writelines(new_contents)

        func_content = "\n".join(func_content)
        return f"Edited function with new content: {func_content}"
    
    def run_experiment(self, **kwargs):
        return self.command_line(f"bash refsol.sh", root_dir=self.workspace_root)

    # Execution
    def execute_python_script(self, file_name, arguments="", **kwargs):
        file_name = file_name.strip()
        if not os.path.exists(os.path.join(self.cur_dir,file_name)):
            return f"The file {file_name} does not exist. Tip: use the file's relative path or change the directory."
        return self.command_line(f"python -u {file_name} {arguments}")

    def execute_bash_script(self, file_name, arguments="", **kwargs):
        file_name = file_name.strip()
        if not os.path.exists(os.path.join(self.cur_dir,file_name)):
            return f"The file {file_name} does not exist. Tip: use the file's relative path or change the directory."
        return self.command_line(f"export MKL_SERVICE_FORCE_INTEL=1 && bash {file_name} {arguments}")

    def command_line(self, command, root_dir=None, **kwargs):
        command = command.strip()
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True, cwd=root_dir if root_dir else self.cur_dir)

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
                        # print("STDOUT:", line, end =" ")
                        stdout_lines.append(line)
                        lines.append(line)
                    else:
                        # print("STDERR:", line, end =" ")
                        stderr_lines.append(line)
                        lines.append(line)

            for line in process.stdout:
                line = line
                # print("STDOUT:", line, end =" ")
                stdout_lines.append(line)
                lines.append(line)
            for line in process.stderr:
                line = line
                # print("STDERR:", line, end =" ")
                stderr_lines.append(line)
                lines.append(line)


            lines = [line for line in lines if "Error: mkl-service + Intel(R)" not in line and "MKL_SERVICE_FORCE_INTEL" not in line]

            return_code = process.returncode

            if return_code != 0:
                observation = "".join(stderr_lines)
            else:
                observation = "".join(stdout_lines)

            if observation == "":
                observation = "".join(lines)
            
            # Tips
            observation = observation.replace("Error: mkl-service + Intel(R) MKL: MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1 library.", "")
            if "FileNotFoundError" in observation or "ModuleNotFoundError" in observation:
                observation += "\nTip: Verify that the file/directory exists.You could be running the script in a wrong directory, If so, try changing directory. Refer to the README for examples."
            return observation
        except Exception as e:
            return f"Something went wrong in executing {command}: {e}."

    # Directory 
    def list_files(self, directory):
        root = os.path.abspath(self.workspace_root)
        directory = os.path.abspath(os.path.join(self.cur_dir, directory))
        if directory.startswith(root) and os.path.exists(directory):
            try:
                return self.command_line(f"ls -F {directory}")
            except Exception as e:
                return f"Cannot list files due to {e}"            
        else:
            return f"Directory not found in the root directory. Tip: use list_files to see contents of the current directory"
        
    def move(self, source, destination, *kwargs):
        try:
            return self.command_line(f"mv {os.path.abspath(os.path.join(self.cur_dir, source))} {os.path.abspath(os.path.join(self.cur_dir, destination))}")
        except Exception as e:
            return f"Cannot move file"

    def change_directory(self, directory):
        root = os.path.abspath(self.workspace_root)
        directory = os.path.abspath(os.path.join(self.cur_dir, directory))
        if directory.startswith(root) and os.path.exists(directory):
            self.cur_dir = directory
            return f"Directory successfully changed to {self.cur_dir[len(root)+1:]}"
        else:
            return f"Directory not found in the root directory. Tip: use list_files to see contents of the current directory"

