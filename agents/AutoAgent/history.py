import wandb
import os
import datetime

this_path = os.path.dirname(__file__)

class History:
    def __init__(self, tags) -> None:
        self.research_plans = []
        self.actions = []
        self.observations = []
        self.logs = []
        self.tags = tags 
        self.timestr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    def append_research_plan(self, rp):
        wandb.log({"research_plan": rp})
        self.research_plans.append(rp)
        self.logs.append({"research_plan": rp})
    
    def append_action(self, action):
        wandb.log({"action": action})
        self.actions.append(action)
        self.logs.append({"action": action})

    def append_observation(self, obs):
        wandb.log({"observation": obs})
        self.observations.append(obs)
        self.logs.append({"observation": obs})
     
    def save_history(self):
        wandb.log({"research_plans": self.research_plans,
                   "actions": self.actions,
                   "observations": self.observations})

        with open(os.path.join(this_path, "logs", f"{self.timestr}_{'_'.join(self.tags)}"), 'w') as file:
            file.write("Logs \n\n")
            step = 0
            for log in self.logs:
                if "research_plan" in log:
                    file.write(f"Step: {step}\n\n")
                    step += 1 
                    file.write(f"Research Plan:\n{log['research_plan']}\n\n")
                elif 'action' in log:
                    file.write(f"Action: {log['action']}\n\n")
                elif 'observation' in log:
                    file.write(f"Observation: \n{log['observation']}\n\n")

            
