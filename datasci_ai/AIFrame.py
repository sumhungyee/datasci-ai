import pandas as pd
from tqdm import tqdm
from datasci_ai import errors

class AIDataFrame(pd.DataFrame):
    def __init__(self, llm, data=None, index=None, columns=None, dtype=None, copy=None):
        super().__init__(data=data, index=index, columns=columns, dtype=dtype, copy=copy)
        self.llm = llm
        self.name = "ddff"
        self.df_details = f"""{self.name} is a pandas dataframe. {self.name} has {self.shape[0]} rows and {self.shape[1]} columns, and the list of its columns are {list(self.columns)}.
Here is a header of {self.name}:\n{self.head().to_string(index=False)}\n"""
        self.prompt = """
Below is an instruction that describes a programming task. Write a response in python that appropriately completes the request in markdown format, using one code block without explanations.   
### Instruction: {df_details} {query}
### Response:
"""

    def request(self, query, verbose=False, addon="", max_iters=5):
        full_prompt = self.prompt.format(df_details=self.df_details, query=query)
        start_token = r"```(python)?\r"
        reply = self.llm.generate_reply(full_prompt, verbose=verbose) # returns a Reply
        exec(f"{self.name} = self.copy()")
        
        full_query = query + "\n" + addon if addon else query

        if max_iters > 0:

            try:
                code = reply.extract_code(start_token=start_token)
            except AttributeError as err:
                tqdm.write(f"\nError encountered: {max_iters-1} tries left. Error: {err}")
                msg = f"You replied with the following response\n{reply.text}\nHowever, the code block was not properly formatted in markdown format."
                self.request(full_query, verbose=verbose, addon=msg, max_iters=max_iters-1)
            except Exception as f:
                language = reply.extract_language()
                raise errors.LanguageError(language)
            try:
                exec(code)
            except Exception as e: 
                tqdm.write(f"\nError encountered: {max_iters-1} tries left. Error: {e}")               
                msg = f"You provided this code:\n{code}\nHowever, the following error was thrown:\n{e.__class__.__name__}: {e}"
                self.request(full_query, verbose=verbose, addon=msg, max_iters=max_iters-1)
            
            return AIDataFrame(self.llm, data=eval(f"{self.name}"))
            



    def copy(self):
        df = super().copy()
        return AIDataFrame(self.llm, data=df)
    


    
    
