class Environment:
    def __init__(self, root_dir, max_steps) -> None:
        self.step = 0
        self.max_steps = max_steps
        self.root_dir = root_dir
        self.cur_dir = root_dir
        self.acis = []

    def execute(self, action):
        if self.step >= self.max_steps:
            return "Environment closed due to maximum steps reached"
        self.step += 1

        observation = ""
        return observation

    def add_aci(self, aci):
        self.acis.append(aci)
