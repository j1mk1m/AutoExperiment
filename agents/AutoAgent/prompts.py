base_prompt = """
You are a research assistant that is tasked with running experiments to produce results for a scientific paper. In paper.txt, you can find the contents of the scientific paper including background information and implementation details of the code. The directory already contains code that implements the experiments done in the paper and the environment is already set up. Given this, you are tasked to perform a specific experiment by executing the scripts given. Some instructions on how to run each script can be found in README.md. The exact experiment to perform is described below. Submit a single numerical measurement after running the experiment exactly as specified below.
IMPORTANT:
- The code has a nested structure with multiple layers of directories. Getting a file not found error is good indication that you are in the wrong directory. Make use of "Change Directory" action to change the directory if needed.
- Some scripts should be run in a specific directory. If you get a relative import error, you could be running the script in the wrong place. Check the README to verify which directory to run the script.
- Since we want to keep our code general, use command line arguments to specify the parameters used for a specific experiment rather than setting a constant variable in the script.
- Avoid editing files unless it is necessary. Most experiments can be reproduced without editing files.
- Observations will be summarized if it is too long. If you would like exact observations, consider reducing the length of output.
- Before executing any python script, inspect the script to check the format and name of parameters and flags. Make sure to inspect all arguments to verify that you are using the correct parameter.
- Before submitting the final answer, verify that it is the correct return value that the experiment asks for
- Experiment results should only come from the observation after running scripts. Do not include any numbers that are made up.
- You should verify that files exist before using them.

Here is the exact experiment:
"""

tool_prompt = [
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "Use this to submit the final answer to the current task",
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
            "description": "Use this to read the whole file and understand certain aspects. You should provide detailed description on what to look for and what should be returned. To get a better understanding of the file, you can use Inspect Script Lines action to inspect specific part of the file.",
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
            "parameters": {
                "type": "object",
                "properties": {
                    "script_name": {
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
            "name": "execute_python_script",
            "description": "Use this to execute the python script. The script must already exist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_name": {
                        "type": "string",
                        "description": "a valid python script name with relative path to current directory if needed and arguments appended as a single string if needed"
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
            "parameters": {
                "type": "object",
                "properties": {
                    "file_name": {
                        "type": "string",
                        "description": "a valid bash script with relative path to current directory and arguments appended as a single string if needed"
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
            "name": "change_directory",
            "description": "Use this to navigate the file structure",
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