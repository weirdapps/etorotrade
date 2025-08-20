"""Console color utilities"""
class ConsoleColors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"
    
    @classmethod
    def colorize(cls, text: str, value: float) -> str:
        if value > 0:
            return f"{cls.GREEN}{text}{cls.RESET}"
        elif value < 0:
            return f"{cls.RED}{text}{cls.RESET}"
        else:
            return f"{cls.YELLOW}{text}{cls.RESET}"
