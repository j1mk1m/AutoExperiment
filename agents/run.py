import wandb

from agents.environment import Environment, MLAgentBench_Env, SWE_AGENT_Env
from agents.agent import Agent, ReActAgent, PlanningAgent, MLAgentBenchAgent
from agents.llm import LLM 
from agents.memory import Memory, FullMemory, SlidingWindowMemory, Summary


def get_env(env_args, llm_manager, X, metadata):
    if env_args.environment == "MLAgentBench":
        return MLAgentBench_Env(env_args.max_compute_time, llm_manager, X, metadata)
    elif env_args.environment == "SWE-Agent":
        return SWE_AGENT_Env(llm_manager, X, metadata, **vars(env_args))
    else:
        raise NotImplementedError()

def get_memory(memory_args, llm_manager):
    if memory_args.memory == "Full":
        return FullMemory(llm_manager)
    elif memory_args.memory == "SlidingWindow":
        return SlidingWindowMemory(llm_manager, lookback=memory_args.lookback)
    elif memory_args.memory == "Summary":
        return Summary(llm_manager, lookback=memory_args.lookback)
    else:
        raise NotImplementedError()

def get_agent(agent_args, env, llm_manager, memory, X, metadata):
    if agent_args.agent == "ReAct":
        return ReActAgent(env, llm_manager, memory, X, metadata, agent_args.max_retries)
    elif agent_args.agent == "Planning":
        return PlanningAgent(env, llm_manager, memory, X, metadata, agent_args.max_retries)
    elif agent_args.agent == "MLAgentBench":
        return MLAgentBenchAgent(env, llm_manager, memory, X, metadata, agent_args.max_retries)
    else:
        raise NotImplementedError()


def run_agent(args, X, metadata, tags):
    # initialize
    llm_manager = LLM(args.model_engine, args.compute_budget) 
    env = get_env(args, llm_manager, X, metadata)
    memory = get_memory(args, llm_manager)
    agent = get_agent(args, env, llm_manager, memory, X, metadata)

    return agent.run(args.max_agent_steps, tags)

    