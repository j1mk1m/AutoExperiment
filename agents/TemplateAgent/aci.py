class ACI:
    def __init__(self, name) -> None:
        self.name = name

    def execute(self, *args):
        pass

class ReadFileACI(ACI):
    def __init__(self, name) -> None:
        super().__init__(name)

    def execute(self, *args):
        return super().execute(*args)