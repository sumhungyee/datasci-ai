class DataSciAIError(Exception):
    def __init__(self, message="datasci_ai encountered an error!"):
        self.message = message
        super.__init__(self.message)

class LanguageError(DataSciAIError):
    def __init__(self, language):
        self.message = f"Expected python, got {language}!"
        super.__init__(self.message)

