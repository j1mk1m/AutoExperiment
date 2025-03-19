import dataclasses
import os
import shutil
import subprocess
import selectors
import difflib
import sys
import datetime 
this_dir = os.path.dirname(__file__)
sys.path.append(this_dir)

from llm import call_llm

def add_env_args(parser):
    parser.add_argument("--environment", type=str, choices=["MLAgentBench", "SWE-Agent"], required=True,
                        help="Type of environment to use")

class ACI:
    def __init__(self, name, description, args, func) -> None:
        self.name = name
        self.description = description
        self.args = args
        self.func = func

@dataclasses.dataclass
class EnvironmentStep:
    action: str
    action_input: dict
    observation: str
    done: bool

class Environment:
    def __init__(self, llm_manager, X, metadata, **kwargs) -> None:
        self.llm_manager = llm_manager
        self.X = X
        self.metadata = metadata
        self.combined_id = metadata["combined_id"]
        self.mode = metadata["mode"]
        self.source = X["path"]

        self.compute_time = 0

        # Experiments
        self.retrieval = kwargs["retrieval"] if "retrieval"  in kwargs else "agent"
        self.remove_code_context = "remove_code_context" in kwargs and kwargs["remove_code_context"]

        self.acis = [
            ACI(name="final_answer", 
                description="Use this to submit the final answer to the current task", 
                args={
                "final_answer": {
                    "type": "string",
                    "description": "json format string representing dictionary of the final answer"
                }}, 
                func=self.final_answer),
            ACI(name="command_line",
                description="Use this to run a linux command",
                args={
                    "command": {
                        "type": "string",
                        "description": "valid linux command line command"
                    }
                }, 
                func=self.command_line)
        ]

    def reset(self):
        self._setup_workspace(self.source)
        self.cur_dir = self.workspace_root
        self.action_to_function_mapper = {aci.name: aci.func for aci in self.acis}
 
    def _setup_workspace(self, source):
        self.workspace_root = os.path.normpath(os.path.join(this_dir, "workspace", f"{self.mode}_{self.combined_id}"))
        if os.path.exists(self.workspace_root):
            shutil.rmtree(self.workspace_root)

        shutil.copytree(source, self.workspace_root, symlinks=True)

    def get_exp_description(self):
        with open(os.path.join(self.source, "experiment.txt"), 'r') as file: 
            return file.read()

    def get_tool_info(self):
        tools = []
        for aci in self.acis:
            tool = {
                "type": "function",
                "function": {
                    "name": aci.name,
                    "description": aci.description,
                    "required": [arg_name for arg_name in aci.args.keys()],
                    "parameters": {
                        "type": "object",
                        "properties": aci.args
                    }
                }
            }
            tools.append(tool)
        return tools

    def get_tool_descriptions(self):
        return "\n".join([f"{aci.name}: {aci.description}" for aci in self.acis])

    def execute(self, action, inputs):
        if action not in self.action_to_function_mapper:
            return f"Tool {action} not supported", False

        start_time = datetime.datetime.now()
        try:
            observation = self.action_to_function_mapper[action](**inputs)
        except Exception as e:
            observation = f"Execution of {action} resulted in error {e}"
        end_time = datetime.datetime.now()

        done = action == "final_answer"

        self.compute_time += int((end_time - start_time).total_seconds())
        return EnvironmentStep(action, inputs, observation, done)

    # Actions
    def final_answer(self, final_answer, **kwargs):
        return final_answer

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


class MLAgentBench_Env(Environment):
    def __init__(self, llm_manager, X, metadata, **kwargs) -> None:
        super().__init__(llm_manager, X, metadata, **kwargs)

        tools = [
            ACI(name="inspect_file_lines", 
                description="Use this to inspect specific part of a file precisely, or the full content for short files.",
                args={
                    "file_name": {
                        "type": "string",
                        "description": "a valid python script name with relative path to current directory if needed"
                    },
                    "start_line_number": {
                        "type": "number",
                        "description": "a valid line number"
                    },
                    "end_line_number": {
                        "type": "number",
                        "description": "a valid line number"
                    }
                },
                func=self.inspect_file_lines),
            ACI(name="read_file",
                description="Use this to read content from an existing file.",
                args={
                    "file_name": {
                        "type": "string", 
                        "description": "a valid file name with relative path to current directory if needed"
                    }
                },
                func=self.read_file),
            ACI(name="write_file",
                description="Use this to write content to a file. If the file does not exist, a new file will be created. If file exists, content will be overriden",
                args={
                    "file_name": {
                        "type": "string", 
                        "description": "a valid file name with relative path to current directory if needed"
                    },
                    "content": {
                        "type": "string",
                        "description": "the content to be written to the file"
                    }
                },
                func=self.write_file),
            ACI(name="append_file",
                description="Use this to append content to an existing file.",
                args={
                    "file_name": {
                        "type": "string", 
                        "description": "a valid file name with relative path to current directory if needed"
                    },
                    "content": {
                        "type": "string",
                        "description": "the content to be appended to the file"
                    }
                },
                func=self.append_file),
            ACI(name="edit_file",
                description="Use this to do a relatively large but cohesive edit over a python script. Instead of editing the script directly, you should describe the edit instruction so that another AI can help you do this.",
                args={
                    "file_name": {
                        "type": "string",
                        "description": "a valid file name with relative path to current directory if needed. An empty file will be created if it does not exist."
                    },
                    "edit_instruction": {
                        "type": "string", 
                        "description": "a detailed step by step description on how to edit it."
                    },
                    "save_name": {
                        "type": "string",
                        "description": "a valid file name with relative path to current directory if needed"
                    }
                },
                func=self.edit_file),
            ACI(name="understand_file",
                description="Use this to read the whole file and understand certain aspects. You can provide a detailed description on what to look for and what should be returned.",
                args={
                    "file_name": {
                        "type": "string",
                        "description": "a valid file name with relative path to current directory if needed"
                    },
                    "things_to_look_for": {
                        "type": "string",
                        "description": "a detailed description on what to look for and what should returned"
                    }
                },
                func=self.understand_file),
            ACI(name="list_files",
                description="Use this to list files in a directory",
                args={
                    "directory": {
                        "type": "string",
                        "description": "valid path to directory"
                    }
                },
                func=self.list_files),
            ACI(name="copy_file", 
                description="Use this to copy a file from source to destination",
                args={
                    "source": {
                        "type": "string",
                        "description": "valid path to file"
                    },
                    "destination": {
                        "type": "string",
                        "description": "valid path to destination"
                    }
                },
                func=self.copy_file),
            ACI(name="change_directory",
                description="Use this to change the current working directory",
                args={
                    "directory": {
                        "type": "string",
                        "description": "valid path to directory"
                    }
                },
                func=self.change_directory),
            ACI(name="execute_python_script",
                description="Use this to execute the python script. The script must already exist.",
                args={
                    "file_name": {
                        "type": "string",
                        "description": "a valid python script name with relative path to current directory if needed"
                    },
                    "arguments": {
                        "type": "string",
                        "description": "command line arguments to use if needed"
                    }
                },
                func=self.execute_python_script),
            ACI(name="execute_bash_script",
                description="Use this to execute a bash script. The script must already exist",
                args={
                    "file_name": {
                        "type": "string",
                        "description": "a valid bash script with relative path to current directory"
                    },
                    "arguments": {
                        "type": "string",
                        "description": "command line arguments to use if needed"
                    }
                },
                func=self.execute_bash_script)
        ]
        self.acis += tools 

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

    def append_file(self, file_name, content, **kwargs):
        with open(os.path.join(self.cur_dir, file_name), 'a') as file:
            file.write(content)
        return "File appended successfully"

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

        completion = self.llm_manager.call_llm([{"role": "system", "content": prompt}], None).response.content

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

    def edit_script_segment(self, file_name, start_line, end_line, edit_instruction, save_name=None, **kwargs):
        try:
            content = self.inspect_file_lines(file_name, start_line, end_line)
            prev_content = self.inspect_file_lines(file_name, 0, start_line)
            next_content = self.inspect_file_lines(file_name, end_line+1, None)
        except:
            self.write_file(file_name, "")
            content = ""
            prev_content = ""
            next_content = ""

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

        completion = self.llm_manager.call_llm([{"role": "system", "content": prompt}], None).response.content

        new_content = completion.strip()
        if "```python" in new_content:
            new_content = new_content.split("```python")[1].split("```")[0].strip()
        if "```bash" in new_content:
            new_content = new_content.split("```bash")[1].split("```")[0].strip()
        if "```" in new_content:
            new_content = new_content.split("```")[1]

        new_content = prev_content + new_content + next_content

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
        return self.command_line(f"export MKL_SERVICE_FORCE_INTEL=1 && bash {file_name} {arguments}")

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
        
    def copy_file(self, source, destination, *kwargs):
        try:
            return self.command_line(f"cp {os.path.abspath(os.path.join(self.cur_dir, source))} {os.path.abspath(os.path.join(self.cur_dir, destination))}")
        except Exception as e:
            return f"Cannot move file. Got error {e}"

    def change_directory(self, directory):
        root = os.path.abspath(self.workspace_root)
        directory = os.path.abspath(os.path.join(self.cur_dir, directory))
        if directory.startswith(root) and os.path.exists(directory):
            self.cur_dir = directory
            return f"Directory successfully changed to {self.cur_dir[len(root)+1:]}"
        else:
            return f"Directory not found in the root directory. Tip: use list_files to see contents of the current directory"


class SWE_AGENT_Env(Environment):
    def __init__(self, llm_manager, X, metadata, **kwargs) -> None:
        super().__init__(llm_manager, X, metadata, **kwargs)

        self.action_to_function_mapper["find_file"] = self.find_file
        self.action_to_function_mapper["search_file"] = self.search_file
        self.action_to_function_mapper["search_dir"] = self.search_dir
        self.action_to_function_mapper["open"] = self.open
        self.action_to_function_mapper["scroll_up"] = self.scroll_up
        self.action_to_function_mapper["scroll_down"] = self.scroll_down
        self.action_to_function_mapper["goto"] = self.goto
        self.action_to_function_mapper["edit"] = self.edit

        self.opened_file = None
        self.line_start = 0
        self.line_end = 100

    # actions
    def find_file(self, filename):
        pass

    def search_file(self, query):
        pass

    def search_dir(self, query):
        pass

    def open(self, file_path):
        pass

    def scroll_up(self):
        pass

    def scroll_down(self):
        pass

    def goto(self, line):
        pass

    def edit(self, start_line, end_line, replacement_text):
        pass



