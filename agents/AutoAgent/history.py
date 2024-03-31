import wandb
import os

this_path = os.path.dirname(__file__)

class History:
    def __init__(self) -> None:
        self.research_plans = []
        self.actions = []
        self.observations = []
        self.memories = []

    def append_research_plan(self, rp):
        wandb.log({"research_plan": rp})
        self.research_plans.append(rp)
    
    def append_action(self, action):
        wandb.log({"action": action})
        self.actions.append(action)

    def append_observation(self, obs):
        wandb.log({"observation": obs})
        self.observations.append(obs)
    
    def append_memory(self, memory):
        wandb.log({"memory": memory})
        self.memories.append(memory)
 
    def save_history(self, mode, paper_id, exp_id):
        wandb.log({"research_plans": self.research_plans,
                   "actions": self.actions,
                   "observations": self.observations,
                   "memories": self.memories})

        with open(os.path.join(this_path, "logs", f"{mode}_{paper_id}_{exp_id}"), 'w') as file:
            file.write("Logs \n\n")
            for i in range(len(self.research_plans)):
                file.write(f"Step {i} \n")
                file.write(f"Research plan: {self.research_plans[i]}")
                file.write("\n\n")
                if (i >= len(self.actions)): break
                file.write(f"Action: {self.actions[i]["action"]} \n Action Inputs: {self.actions[i]["arguments"]}")
                file.write("\n\n")
                if (i >= len(self.observations)): break
                file.write(f"Observation: {self.observations[i]}")
                file.write("\n\n")
                if (i >= len(self.memories)): break
                file.write(f"Memory: {self.memories[i]}")
                file.write("\n\n")

