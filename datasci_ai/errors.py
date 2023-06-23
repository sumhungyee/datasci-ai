class DataSciAIError(Exception):
    def __init__(self, message="datasci_ai encountered an error!"):
        self.message = message
        super.__init__(message)