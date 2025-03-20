import wandb

from agents.environment import Environment, MLAgentBench_Env, SWE_AGENT_Env
from agents.agent import Agent, ReActAgent, MLAgentBenchAgent
from agents.llm import LLM 
from agents.memory import Memory, FullMemory, SlidingWindowMemory
import agents.logger as logger


def get_env(env_args, llm_manager, X, metadata):
    if env_args.environment == "MLAgentBench":
        return MLAgentBench_Env(llm_manager, X, metadata, **vars(env_args))
    elif env_args.environment == "SWE-Agent":
        return SWE_AGENT_Env(llm_manager, X, metadata, **vars(env_args))
    else:
        raise NotImplementedError()

def get_memory(memory_args):
    if memory_args.memory == "Full":
        return FullMemory()
    elif memory_args.memory == "SlidingWindow":
        return SlidingWindowMemory(lookback=memory_args.lookback)

def get_agent(agent_args, env, llm_manager, memory, X, metadata):
    if agent_args.agent == "ReAct":
        return ReActAgent(env, llm_manager, memory, X, metadata, agent_args.max_retries)
    elif agent_args.agent == "MLAgentBench":
        return MLAgentBenchAgent(env, llm_manager, memory, X, metadata, agent_args.max_retries)
    else:
        raise NotImplementedError()


def run_agent(args, X, metadata, tags):
    # initialize
    llm_manager = LLM(args.model_engine) 
    env = get_env(args, llm_manager, X, metadata)
    env.reset()
    memory = get_memory(args)
    agent = get_agent(args, env, llm_manager, memory, X, metadata)

    logfile = logger.create_log(tags)

    for i in range(args.max_agent_steps):
        print("###############################")
        print(f"Step {i}\n")
        is_last_step = llm_manager.cost >= args.compute_budget or env.compute_time >= args.max_compute_time or i == args.max_agent_steps - 1

        action, inputs = agent.step(is_last_step)

        if action is None:
            return "Agent did not provide a valid action response"

        step = env.execute(action, inputs)
        agent.add_observation(step.observation)
        print(f"### Observation ###\n{step.observation}\n")
        logger.write_log(logfile, agent.system_prompt, agent.memory, llm_manager, env)
        wandb.log({"step": i, 
                   "compute_time": env.compute_time, 
                   "cost": llm_manager.cost, 
                   "completion_tokens": llm_manager.completion_tokens, 
                   "prompt_tokens": llm_manager.prompt_tokens})

        if step.done:
            return step.observation # return final answer

        if is_last_step:
            message = "Failed due to "
            if llm_manager.cost >= args.compute_budget:
                message += "exceeding LLM compute budget"
            elif env.compute_time >= args.max_compute_time:
                message += "exceeding maximum compute time"
            else:
                message += "exceeding maximum number of agent steps"
            return message
    
    return "Failed due to exceeding maximum number of agent steps"
