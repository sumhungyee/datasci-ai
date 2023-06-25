import pandas as pd
import matplotlib.pyplot as plt
from datasci_ai.errors import LanguageError, CodeDetectionError, CodeGenerationError

class AIDataFrame(pd.DataFrame):
    def __init__(self, llm, data=None, index=None, columns=None, dtype=None, copy=None):
        super().__init__(data=data, index=index, columns=columns, dtype=dtype, copy=copy)
        self.llm = llm
        self.name = "tempdf"
        self.df_details = f"""You are provided a pandas dataframe stored in the variable {self.name}. {self.name} has {self.shape[0]} rows and {self.shape[1]} columns, and the here is a dictionary of its columns and their datatypes:{dict(self.dtypes)}.\n"""
        self.prompt = """
{df_details}Below is an instruction that describes a programming task. Write a response that appropriately completes the request and provide code to perform some operations on this existing DataFrame {name}. Write in markdown format and use only one code block.  
### Instruction: {query}
### Response:
"""

    def request(self, query, verbose=True, addon="", max_iters=5):
        full_query = query + "\n" + addon if addon else query
        full_prompt = self.prompt.format(df_details=self.df_details, query=full_query, name=self.name)
        start_token = r"```(python)?\r"
        reply = self.llm.generate_reply(full_prompt, verbose=verbose) # returns a Reply
        exec(f"{self.name} = self.copy()")
        
        
        if max_iters > 0:
            try:
                code = reply.extract_code(start_token=start_token)
            except AttributeError as err:
                language = reply.extract_language()
                if language.lower() != "python" and "python" not in language.lower(): # If language isn't python
                    raise LanguageError(language) from None
                else:
                    print(f"\nError encountered: {max_iters-1} tries left. Error: {err}")
                    msg = f"You replied with the following response\n{reply.text}\nHowever, the code block was not properly formatted in markdown format."
                    return self.request(query, verbose=verbose, addon=msg, max_iters=max_iters-1)

            except Exception as f:
                print(f"\nError detecting code! {max_iters-1} tries left.")
                return self.request(query, verbose=verbose, max_iters=max_iters-1)
            
            try:
                exec(code)
            except FileNotFoundError as fe:
                print(f"\nError encountered: {max_iters-1} tries left. Erroneous rereading of data detected.")
                msg = f"You provided this code:\n{code}\nHowever, the following error was thrown:\n{fe.__class__.__name__}: {fe}\n {self.name} already contains data, and there is no need to use pd.read_csv(). Correct these errors, writing code in markdown format using one code block."
                plt.close('all')
                return self.request(query, verbose=verbose, addon=msg, max_iters=max_iters-1)
            
            except Exception as e: 
                print(f"\nError encountered: {max_iters-1} tries left. Error: {e}")               
                msg = f"You provided this code:\n{code}\nHowever, the following error was thrown:\n{e.__class__.__name__}: {e}\nCorrect these errors, writing code in markdown format using one code block."
                plt.close('all')
                return self.request(query, verbose=verbose, addon=msg, max_iters=max_iters-1)
            
            return AIDataFrame(self.llm, data=eval(f"{self.name}"))
        
        else:
            try:
                code = reply.extract_code(start_token=start_token)
            except Exception:
                language = reply.extract_language()
                if language.lower() != "python" and "python" not in language.lower(): # If language isn't python
                    raise LanguageError(language) from None
                raise CodeDetectionError() from None
            try:
                exec(code)
            except Exception as ex:
                raise CodeGenerationError(ex) from None
            
            return AIDataFrame(self.llm, data=eval(f"{self.name}"))

        
        
            
    
    def copy(self, deep=True):
        df = super().copy(deep=deep)
        return AIDataFrame(self.llm, data=df)
    


    
    
