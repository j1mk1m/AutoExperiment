import datetime
import os
this_path = os.path.dirname(__file__)

def create_log(tags):
    timestr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logfile = os.path.join(this_path, "logs", f"{timestr}_{'_'.join(tags)}.txt")
    print(f"Log file created at {logfile}\n")
    return logfile

def write_log(logfile, system_prompt, memory, llm_manager, env):
    with open(logfile, 'w') as file:
        file.write("SYSTEM PROMPT\n")
        file.write(system_prompt)
        file.write("\n\n")
        for step in range(len(memory.observations)):
            file.write("######################################\n")
            file.write(f"Step: {step}\n\n")
            file.write(f"Thought:\n{memory.thoughts[step]}\n\n")
            file.write(f"Action:\n{str(memory.tool_calls[step].tool_calls[0].function)}\n\n")
            file.write(f"Observation:\n{memory.observations[step]['content']}\n\n")
        
        file.write("######################################\n")
        file.write(f"Compute Cost: {llm_manager.cost} \n")
        file.write(f"Prompt tokens: {llm_manager.prompt_tokens} \n")
        file.write(f"Completion tokens: {llm_manager.completion_tokens} \n")
        file.write(f"Compute Time: {env.compute_time} \n")
        file.write("######################################\n")
