base_prompt = """
You are a research assistant that is tasked with running experiments to produce results for a scientific paper. In paper.txt, you can find the contents of the scientific paper including background information and implementation details of the code. The directory already contains code that implements the experiments done in the paper and the environment is already set up. Given this, you are tasked to perform a specific experiment by executing the scripts given. Some instructions on how to run each script can be found in README.md. The exact experiment to perform is described below. Submit a single numerical measurement after running the experiment exactly as specified below.
Here is the exact experiment:
"""

base_prompt_PC = """
You are a research assistant that is tasked with running experiments to produce results for a scientific paper. In paper.txt, you can find the contents of the scientific paper including background information and implementation details of the code. The directory already contains some code that implements the experiments done in the paper and the environment is already set up. Given this, you are tasked to perform a specific experiment by adding missing code and executing code to get experiment results. Some instructions on how to run each script can be found in README.md. The exact experiment to perform is described below. Submit a single numerical measurement after running the experiment exactly as specified below.
Here is the exact experiment:
"""

base_prompt_NC = """
You are a research assistant that is tasked with running experiments to produce results for a scientific paper. In paper.txt, you can find the contents of the scientific paper including background information and implementation details of the code. The directory contains utility files and dataset. Also, some code structure is given and the environment is already set up. Given this, you are tasked to perform a specific experiment by writing necessary code following the structure of functions given and executing code to get experiment results. Some instructions on how to run each script can be found in README.md. The exact experiment to perform is described below. Submit a single numerical measurement after running the experiment exactly as specified below.
Here is the exact experiment:
"""

rp_prompt = """
The current research plan is {plan}.
Generate a new research plan with current status and confirmed results of each step briefly annotated.
Tip: 
- based on the experiment details, navigate the code base and identify the script you need to run
- refer to README.md on how to run scripts
- before executing python or bash files, inspect file lines to verify parameter names and values
- final answer should be obtained only by executing scripts
"""

repeat_prompt = """
Again, you are a research assistant tasked with running this experiment:
{experiment}

Tips: 
- prefer command line arguments over editing constants in scripts
- avoid editing files unless necessary
- do not repeat the same action unnecessarily
"""

tool_prompt = [
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "Use this to submit the final answer to the current task",
            "required": ["final_answer"],
            "parameters": {
                "type": "object",
                "properties": {
                    "final_answer": {
                        "type": "number",
                        "description": "numeric value of the final experiment result"
                    }
                },
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "understand_file",
            "description": "Use this to read the whole file and understand certain aspects. You can provide a detailed description on what to look for and what should be returned.",
            "required": ["file_name"],
            "parameters": {
                "type": "object",
                "properties": {
                    "file_name": {
                        "type": "string",
                        "description": "a valid file name with relative path to current directory if needed",
                    },
                    "things_to_look_for": {
                        "tupe": "string",
                        "description": "a detailed description on what to look for and what should returned"
                    }
                }
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_file_lines",
            "description": "Use this to inspect specific part of a file precisely, or the full content for short files.",
            "required": ["file_name"],
            "parameters": {
                "type": "object",
                "properties": {
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
                }
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Use this to do a relatively large but cohesive edit over a python script. Instead of editing the script directly, you should describe the edit instruction so that another AI can help you do this.",
            "required": ["file_name", "edit_instructions"],
            "parameters": {
                "type": "object",
                "properties": {
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
                }
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Use this to write content to a file. If the file does not exist, a new file will be created. If file exists, content will be overriden",
            "required": ["file_name", "content"],
            "parameters": {
                "type": "object",
                "properties": {
                    "file_name": {
                        "type": "string",
                        "description": "a valid file name with relative path to current directory if needed"
                    },
                    "content": {
                        "type": "string",
                        "description": "the content to be written to the file"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_python_script",
            "description": "Use this to execute the python script. The script must already exist.",
            "required": ["file_name"],
            "parameters": {
                "type": "object",
                "properties": {
                    "file_name": {
                        "type": "string",
                        "description": "a valid python script name with relative path to current directory if needed"
                    },
                    "arguments": {
                        "type": "string",
                        "description": "command line arguments to use if needed"
                    }
                }
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_bash_script",
            "description": "Use this to execute a bash script. The script must already exist",
            "required": ["file_name"],
            "parameters": {
                "type": "object",
                "properties": {
                    "file_name": {
                        "type": "string",
                        "description": "a valid bash script with relative path to current directory"
                    },
                    "arguments": {
                        "type": "string",
                        "description": "command line arguments to use if needed"
                    }
                }
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "command_line",
            "description": "Use this to run any linux command line command",
            "required": ["command"],
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "valid linux command line command"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "Use this to list files in a directory",
            "required": ["directory"],
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "valid path to directory"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "move",
            "description": "Use this to move a file or directory from source to destination. You can also rename files with this function",
            "required": ["source", "destination"],
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "valid path to file or directory"
                    },
                    "destination": {
                        "type": "string",
                        "description": "valid path to file or directory"
                    }

                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "change_directory",
            "description": "Use this to navigate the file structure",
            "required": ["directory"],
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "valid path to directory"
                    }
                }
            },
        }
    },
]
