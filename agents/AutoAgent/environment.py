import os
import shutil
import subprocess
import selectors
import difflib
import sys
this_dir = os.path.dirname(__file__)
sys.path.append(this_dir)

from llm import call_llm

class Environment:
    def __init__(self, paper_id, exp_id, mode, source, model, **kwargs) -> None:
        self.paper_id = paper_id
        self.exp_id = exp_id
        self.mode = mode
        self.source = source
        self.model = model
        self.setup_workspace(source)
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
            "command_line": self.command_line
        }

    def setup_workspace(self, source):
        self.workspace_root = os.path.normpath(os.path.join(this_dir, "workspace", f"{self.mode}_{self.paper_id}_{self.exp_id}"))
        if os.path.exists(self.workspace_root):
            shutil.rmtree(self.workspace_root)

        shutil.copytree(os.path.join(source, "code"), self.workspace_root, symlinks=True)

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
        Here is a detailed description on what to look for and what should returned: {things_to_look_for}
        The description should short and also reference crtical lines in the script relevant to what is being looked for. Only describe what is objectively confirmed by the file content. Do not include guessed numbers. If you cannot find the answer to certain parts of the request, you should say "In this segment, I cannot find ...".
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

    def edit_file(self, file_name, edit_instruction, save_name=None, **kwargs):
        try:
            content = self.read_file(file_name)
        except:
            self.write_file(file_name, "")
            content = ""
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
        if "```" in new_content:
            new_content = new_content.split("```")[1].strip()

        if save_name is None: save_name = file_name
        self.write_file(save_name, new_content)

        diff = list(difflib.unified_diff(content.splitlines(keepends=True), new_content.splitlines(keepends=True)))
        diff = "".join(diff)

        return f"The edited file is saved to {save_name}. Here is the diff, please check if the edit is correct and desirable:\n\n" + diff

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
        return self.command_line(f"bash -u {file_name} {arguments}")

    def command_line(self, command, **kwargs):
        command = command.strip()
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True, cwd=self.cur_dir)

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

            return_code = process.returncode

            if return_code != 0:
                observation = "".join(lines)
            else:
                observation = "".join(lines)
            if observation == "" and return_code == 0:
                # printed to stderr only
                observation = "".join(lines)
            
            # Tips
            if "FileNotFoundError" in observation or "ModuleNotFoundError" in observation:
                observation += "\nTip: Verify that the file/directory exists.You could be running the script in a wrong directory, If so, try changing directory. Refer to the README for examples."
            return observation
        except Exception as e:
            return f"Something went wrong in executing {command}: {e}."

    # Directory 
    def list_files(self, directory):
        try:
            return self.command_line(f"ls -F {os.path.abspath(os.path.join(self.cur_dir, directory))}")
        except Exception as e:
            return f"Cannot list files due to {e}"            
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

