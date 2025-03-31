from agent import Agent

class Agentless(Agent):
    def __init__(self, env, llm_manager, memory, X, metadata, max_retries=3) -> None:
        super().__init__(env, llm_manager, memory, X, metadata, max_retries)

        self.thought_prompt = ""
        self.thought_reprompt = ""

    def _retrieve_nl(self):
        pass

    def _retrieve_code(self):
        pass
    
    def run(self, max_agent_steps, tags):
        self.env.setup()
        logfile = logger.create_log(tags)

        paper_context = self._retrieve_nl()
        code_context = self._retrieve_code()
        thought = f"### Paper Context \n{paper_context}\n\n### Code Context \n{code_context}"

        prompt = f"Paper {paper_context} code {code_context}"

        llm_response = self.llm_manager.call_llm([{"role": "user", "content": prompt}], None)

        self.env.write_file()

        observation = self.env.run_bash()

        # extract
        llm_response = self.llm_manager.call_llm([{"role": "user", "content": f"Given observation, extract results {observation}"}], None)

        return llm_response.response.content


    
