import wandb

from autoexperiment.agents.environment import Environment, BasicEnvironment 
from autoexperiment.agents.agent import Agent, ReActAgent, PlanningAgent, MLAgentBenchAgent
from autoexperiment.agents.agentless import Agentless
from autoexperiment.agents.llm import LLM 
from autoexperiment.agents.memory import Memory, FullMemory, SlidingWindowMemory, Summary


def get_env(env_args, llm_manager, X, metadata, tags):
    return BasicEnvironment(llm_manager, X, metadata, tags, **vars(env_args))
   
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
        return ReActAgent(env, llm_manager, memory, X, metadata, **vars(agent_args))
    elif agent_args.agent == "Planning":
        return PlanningAgent(env, llm_manager, memory, X, metadata, **vars(agent_args))
    elif agent_args.agent == "MLAgentBench":
        return MLAgentBenchAgent(env, llm_manager, memory, X, metadata, **vars(agent_args))
    elif agent_args.agent == "Agentless":
        return Agentless(env, llm_manager, memory, X, metadata, **vars(agent_args))
    else:
        raise NotImplementedError()


def run_agent(args, X, metadata, tags):
    # initialize
    if args.agent == "Agentless":
        args.compute_budget = None
    
    llm_manager = LLM(args.model_engine, args.compute_budget) 
    env = get_env(args, llm_manager, X, metadata, tags)
    memory = get_memory(args, llm_manager)
    agent = get_agent(args, env, llm_manager, memory, X, metadata)

    return agent.run(args.max_agent_steps, tags)

    