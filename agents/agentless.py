from agents.agent import Agent
from agents.retrieval import CodeSearchEngine, SearchEngine
import wandb

from agents.llm import LLM

class Agentless(Agent):
    def __init__(self, env, llm_manager, memory_module, X, metadata, **kwargs) -> None:
        super().__init__(env, llm_manager, memory_module, X, metadata, **kwargs)

        self.max_completion_tokens = kwargs["max_completion_tokens"]

    def _retrieve_nl(self, query):
        text_search = SearchEngine(self.X["path"])
        _, paper_context = text_search.search(query, top_k=2)
        return paper_context

    def _retrieve_code(self, query):
        code_search = CodeSearchEngine(self.X["path"], self.llm_manager)
        _, code_context = code_search.search(query, top_k=2)
        return code_context
    
    def run(self, max_agent_steps, tags):
        total_cost = 0
        self.env.reset()

        func_details = self.X["funcs_to_block"][0]
        file_name = func_details["file"]
        file_content = self.env.read_file(file_name)

        header_line = func_details["header_line"]
        end_line = func_details["line_end"]

        function_content = "\n".join(file_content.split("\n")[header_line-1:end_line-1])

        # Retrieval
        code_context = self._retrieve_code(function_content)
        paper_context = self._retrieve_nl(function_content)

        # Thinking
        prompt = f""" 
You are a helpful coding assistant. You are given contents of a Python file with one missing function. Paper context contains snippets of a research paper that describes how to implement the code. Code context contains code snippets that are similar to the missing function.
### Paper Context ###
{paper_context}

### Code Context ###
{code_context}

### File Content ###
```python
{file_content}
```

Think about how you want to implement the missing Python function.
"""
        print(f"### THOUGHT PROMPT ###\n{prompt}")
        reasoning_tokens = 0
        thought = ""
        while reasoning_tokens < self.max_completion_tokens:
            llm_response = self.llm_manager.call_llm([{"role": "user", "content": prompt}], None, max_completion_tokens=self.max_completion_tokens-reasoning_tokens)

            if llm_response.error:
                self.env.cleanup()
                wandb.log({"compute_cost": total_cost})
                return f"Calling llm for thought failed with {llm_response.response}"
            response = llm_response.response.content + "Wait" # budget forcing

            thought += response
            prompt += response
            reasoning_tokens += llm_response.completion_tokens
            total_cost += llm_response.cost

        wandb.log({"reasoning_tokens": reasoning_tokens}) 
        print(f"### THOUGHT (Cost: {total_cost}) ###\n{thought}")

        # Generate code
        write_prompt = f"""
### Thought ###
{thought}

### File Content ###
```python
{file_content}
```

### Python function ###
```python
{function_content}
```

Give only the missing function implementation. Provide only the code in markdown format. I.e. ```python
"""

        llm_response = self.llm_manager.call_llm([{"role": "system", "content": "You are a helpful coding assistant tasked to fill in a missing Python function. You have previously thought of a strategy."}, 
                                                    {"role": "user", "content": prompt}], None, model="gpt-4o")
        if llm_response.error:
            self.env.cleanup()
            wandb.log({"compute_cost": total_cost})
            return f"Calling llm for code generation failed with {llm_response.response}"

        # Write to file
        response = llm_response.response.content
        total_cost += llm_response.cost
        print(f"### RESPONSE (Cost: {llm_response.cost}) ###\n{response}")
        try:
            new_func_body = response.split("```python")[1].split("```")[0].split("\n")

            new_file_content = file_content.split("\n")[:header_line-1] + new_func_body + file_content.split("\n")[end_line-1:]
            func_details["line_end"] = header_line + len(new_func_body)

            self.env.write_file(file_name, "\n".join(new_file_content)) 
        except Exception as e:
            self.env.cleanup()
            wandb.log({"compute_cost": total_cost})
            return f"Calling llm for code generation failed because of error {e}"

        # run refsol
        observation = self.env.execute_bash_script("refsol.sh")
        print(f"### OBSERVATION ###\n{observation}")

        # extract
        experiment = self.env.get_exp_description()
        prompt = f"""You are a helpful research assistant tasked to report results for an experiment.
Here are the experiment results you need:
{experiment}

Here is the output:
{observation}

Extract the results from the output. Make sure to return a JSON string in the format specified inside ```json.
Format:
```json
json string
```
"""

        llm_response = self.llm_manager.call_llm([{"role": "user", "content": prompt}], None, model="gpt-4o")
        if llm_response.error:
            self.env.cleanup()
            wandb.log({"compute_cost": total_cost})
            return f"Calling llm to extract results failed with {llm_response.response}"
        
        response = llm_response.response.content
        total_cost += llm_response.cost
        print(f"### EXTRACTION (Cost: {llm_response.cost}) ###\n{response}")

        try:
            response = response.split("```json")[1].split("```")[0].strip()
        except Exception as e:
            self.env.cleanup()
            wandb.log({"compute_cost": total_cost})
            return f"Got error extracting json string ouput {e}"
        wandb.log({"compute_cost": total_cost})

        self.env.cleanup()
        return response

    def _is_valid_thought(self, thought):
        pass


    
