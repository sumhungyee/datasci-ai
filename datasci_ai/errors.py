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

class CodeGenerationError(DataSciAIError):
    def __init__(self, error):
        self.message = f"The generated code produces an error: {str(error)}"
        super().__init__(self.message)

class IllegalLoadingError(DataSciAIError):
    def __init__(self):
        self.message = "Illegal loading of data! Data should not be loaded through requests."
        super().__init__(self.message)

class IllegalCodeError(DataSciAIError):
    def __init__(self, name, function=True):
        if function:
            self.message = f"Execution of function '{name}' is restricted."
        else:
            self.message = f"Access to module '{name}' is restricted."
        super().__init__(self.message)