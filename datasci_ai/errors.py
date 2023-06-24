class DataSciAIError(Exception):
    def __init__(self, message="datasci_ai encountered an error!"):
        self.message = message
        super().__init__(self.message)

class LanguageError(DataSciAIError):
    def __init__(self, language):
        self.message = f"Expected python, got {language}!"
        super().__init__(self.message)

class CodeDetectionError(DataSciAIError):
    def __init__(self):
        self.message = "No code blocks detected! Cannot execute code!"
        super().__init__(self.message)
