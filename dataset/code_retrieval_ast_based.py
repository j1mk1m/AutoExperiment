import ast
import os
import re
import json

this_dir = os.path.dirname(__file__)

import ast
import os
import inspect
import importlib.util
import sys
from collections import defaultdict

class FunctionCallVisitor(ast.NodeVisitor):
    def __init__(self):
        self.function_calls = []
        
    def visit_Call(self, node):
        # Check if it's a function call
        if isinstance(node.func, ast.Name):
            self.function_calls.append(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            # Handle imported functions like module.function()
            if isinstance(node.func.value, ast.Name):
                self.function_calls.append(f"{node.func.value.id}.{node.func.attr}")
        self.generic_visit(node)

class FunctionDefVisitor(ast.NodeVisitor):
    def __init__(self):
        self.functions = {}  # Maps function names to their AST nodes
        self.imports = {}    # Maps imported names to their original names/modules
        self.source_lines = None
        
    def set_source_lines(self, source_lines):
        self.source_lines = source_lines
        
    def visit_FunctionDef(self, node):
        if self.source_lines:
            # Store the function node and its source code
            start_line = node.lineno - 1  # AST line numbers are 1-indexed
            end_line = node.end_lineno if hasattr(node, 'end_lineno') else self.find_end_line(start_line)
            source_code = ''.join(self.source_lines[start_line:end_line])
            self.functions[node.name] = {
                'node': node,
                'source': source_code,
                'start_line': start_line + 1,
                'end_line': end_line
            }
        else:
            self.functions[node.name] = {
                'node': node,
                'source': None
            }
        self.generic_visit(node)
        
    def find_end_line(self, start_line):
        # Simple heuristic - for older Python versions without end_lineno
        # This is a fallback and might not be 100% accurate
        indent_level = None
        for i, line in enumerate(self.source_lines[start_line:]):
            if i == 0:
                # Determine the indentation level of the function
                spaces = len(line) - len(line.lstrip())
                indent_level = spaces
                continue
                
            spaces = len(line) - len(line.lstrip())
            # If we find a line with same or less indentation, it's the end
            if line.strip() and spaces <= indent_level:
                return start_line + i
        
        # If we couldn't find the end, return the last line
        return len(self.source_lines)
        
    def visit_Import(self, node):
        for alias in node.names:
            self.imports[alias.asname or alias.name] = alias.name
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node):
        module = node.module
        for alias in node.names:
            imported_name = alias.name
            as_name = alias.asname or imported_name
            self.imports[as_name] = f"{module}.{imported_name}"
        self.generic_visit(node)

def analyze_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        source_lines = content.splitlines(True)  # Keep line endings
    
    # Parse the file
    tree = ast.parse(content, filename=file_path)
    
    # Find all function definitions and imports
    def_visitor = FunctionDefVisitor()
    def_visitor.set_source_lines(source_lines)
    def_visitor.visit(tree)
    
    # Analyze each function for calls
    function_calls = {}
    for func_name, func_info in def_visitor.functions.items():
        call_visitor = FunctionCallVisitor()
        call_visitor.visit(func_info['node'])
        function_calls[func_name] = call_visitor.function_calls
    
    return {
        'functions': def_visitor.functions, 
        'function_calls': function_calls,
        'imports': def_visitor.imports,
        'source_file': file_path
    }

def analyze_project(directory):
    all_files = {}
    all_functions = {}
    all_function_calls = {}
    imports_map = {}
    module_map = {}  # Maps module names to file paths
    
    # Analyze all Python files in the directory
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, directory)
                module_name = os.path.splitext(relative_path)[0].replace(os.path.sep, '.')
                
                try:
                    result = analyze_file(file_path)
                    
                    # Store file-specific data
                    all_files[relative_path] = result
                    module_map[module_name] = relative_path
                    
                    # Combine data across files
                    for func_name, func_info in result['functions'].items():
                        # Use file:function format for unique identification
                        full_name = f"{relative_path}:{func_name}"
                        all_functions[full_name] = func_info
                    
                    for func_name, calls in result['function_calls'].items():
                        full_name = f"{relative_path}:{func_name}"
                        all_function_calls[full_name] = calls
                    
                    imports_map[relative_path] = result['imports']
                    
                except Exception as e:
                    print(f"Error analyzing {file_path}: {e}")
    
    return {
        'files': all_files,
        'functions': all_functions,
        'function_calls': all_function_calls,
        'imports': imports_map,
        'modules': module_map
    }

def resolve_module_path(project_data, module_name, current_file):
    """Attempt to resolve a module name to a file path."""
    # Check if it's a direct module
    if module_name in project_data['modules']:
        return project_data['modules'][module_name]
    
    # Check if it's imported in the current file
    if current_file in project_data['imports']:
        imports = project_data['imports'][current_file]
        if module_name in imports:
            imported_name = imports[module_name]
            
            # If it's a module.function format
            if '.' in imported_name:
                module_part = imported_name.split('.')[0]
                if module_part in project_data['modules']:
                    return project_data['modules'][module_part]
    
    return None

def get_function_calls_with_content(project_data, target_file, target_function): 
    # Get the calls made by the target function
    target_full_name = f"{target_file}:{target_function}"
    
    if target_full_name not in project_data['function_calls']:
        return f"Function {target_function} not found in {target_file}", {}
    
    calls = project_data['function_calls'][target_full_name]
    
    # Collect called function content
    called_functions = {}
    
    for call in calls:
        if "self." in call:
            call = call[5:]
        # Handle simple function names
        if '.' not in call:
            # First, look in the same file
            same_file_func = f"{target_file}:{call}"
            if same_file_func in project_data['functions']:
                called_functions[call] = {
                    'file': target_file,
                    'source': project_data['functions'][same_file_func]['source']
                }
                continue
            
            # Look in other files
            for func_key, func_info in project_data['functions'].items():
                file, func_name = func_key.split(':', 1)
                if func_name == call:
                    called_functions[call] = {
                        'file': file,
                        'source': func_info['source']
                    }
                    break
        else:
            # Handle module.function style calls
            module_name, func_name = call.split('.', 1)
            
            # Try to resolve the module to a file
            module_file = resolve_module_path(project_data, module_name, target_file)
            
            if module_file:
                module_func = f"{module_file}:{func_name}"
                if module_func in project_data['functions']:
                    called_functions[call] = {
                        'file': module_file,
                        'source': project_data['functions'][module_func]['source']
                    }
    
    # Get the source of the original function
    target_source = ""
    if target_full_name in project_data['functions']:
        target_source = project_data['functions'][target_full_name]['source']
    
    return target_source, calls, called_functions

# Example usage
if __name__ == "__main__":
    paper_ids = ["2309.05569", "2303.11932", "2110.03485", "2205.00048"]
    for paper_id in paper_ids:
        project_dir = os.path.join(this_dir, "MLRC", paper_id)
        code_dir = os.path.join(project_dir, "code")
    
        project_data = analyze_project(code_dir)

        # Read functions.jsonl file
        functions_file = os.path.join(project_dir, "functions.jsonl")
        functions = []
        if os.path.exists(functions_file):
            with open(functions_file, 'r') as f:
                for line in f:
                    functions.append(json.loads(line))
            # print(f"Loaded {len(functions)} functions from {paper_id}")
        
        for function in functions:
            target_function = function["name"]
            target_file = function["file"]
        
            # Find all functions called by a specific function
            target_source, called_functions, called_functions_with_content = get_function_calls_with_content(project_data, target_file, target_function)
            print(f"Target function: {target_function}")
            print(f"Functions called by {target_function}: {called_functions_with_content.keys()}")
            
            code_context = ""
            for func_name, info in called_functions_with_content.items():
                code_context += f"\n---- {func_name} (from {info['file']}) ----\n"
                code_context += info['source'] + "\n\n"
            
            function["code_context"] = code_context


        # Write functions back to functions.jsonl
        with open(functions_file, 'w') as f:
            for function in functions:
                json.dump(function, f)
                f.write('\n')
        print(f"Wrote {len(functions)} functions back to {functions_file}")
    
    
