from agents.agent import Agent
from agents.retrieval import CodeSearchEngine, SearchEngine

from agents.llm import LLM

class Agentless(Agent):
    def __init__(self, env, llm_manager, memory, X, metadata, **kwargs) -> None:
        super().__init__(env, llm_manager, memory, X, metadata, **kwargs)

        self.reasoning_effort = kwargs["reasoning_effort"]

    def _retrieve_nl(self, query):
        text_search = SearchEngine(self.X["path"])
        _, paper_context = text_search.search(query, top_k=5)
        return paper_context

    def _retrieve_code(self, query):
        code_search = CodeSearchEngine(self.X["path"], self.llm_manager)
        _, code_context = code_search.search(query, top_k=5)
        return code_context
    
    def run(self, max_agent_steps, tags):
        self.env.setup()

        func_details = self.X["funcs_to_block"][0]
        file_name = func_details["file"]
        file_content = self.env.read_file(file_name)

        header_line = func_details["header_line"]
        end_line = func_details["line_end"]

        function_content = "\n".join(file_content.split("\n")[header_line-1:end_line-1])

        # Retrieval
        paper_context = self._retrieve_nl(function_content)
        code_context = self._retrieve_code(function_content)

        # Thinking
        prompt = f"""
### Paper Context ###
{paper_context}

### Code Context ###
{code_context}

### Python function ###
```python
{function_content}
```

Think for maximum number of tokens how you want to implement this Python function.
"""
        llm_response = self.llm_manager.call_llm([{"role": "user", "content": prompt}], None, reasoning_effort=self.reasoning_effort)
        if llm_response.error:
            return f"Calling llm for thought failed with {llm_response.response}"

        thought = llm_response.response.content

        # Generate code
        write_prompt = f"""
{thought}

### Python function ###
```python
{function_content}
```

Please only output the edited version of this Python function inside ```python environment. Make sure to match the indentation of the current code.

### Edited Python function ###
"""

        llm_response = self.llm_manager.call_llm([{"role": "user", "content": prompt}], None)
        if llm_response.error:
            return f"Calling llm for code generation failed with {llm_response.response}"

        # Write to file
        response = llm_response.response.content
        new_func_body = response.split("```python")[1].split("```")[0].split("\n")

        new_file_content = file_content.split("\n")[:header_line-1] + new_func_body + file_content.split("\n")[end_line-1:]
        func_details["line_end"] = header_line + len(new_func_body)

        self.env.write_file(file_name, "\n".join(new_file_content)) 

        # run refsol
        observation = self.env.run_bash("refsol.sh")

        # extract
        experiment = self.env.get_exp_description()
        prompt = f"""You are a helpful research assistant tasked to report results for an experiment.
Here are the experiment results you need:
{experiment}

Here is the output:
{observation}

Extract the results from the output. Make sure to return a JSON string in the format specified.
"""

        llm_response = self.llm_manager.call_llm([{"role": "user", "content": prompt}], None, model="gpt-4o")
        if llm_response.error:
            return f"Calling llm to extract results failed with {llm_response.response}"

        return llm_response.response.content


    
