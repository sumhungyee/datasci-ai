

from datasci_ai import model, errors
import pandas as pd
import numpy as np
import re
import unittest


class ModelTest(unittest.TestCase):
    def setUp(self):
        self.llm = model.LLM(path="C:\WizardCoder-15B-1.0.ggmlv3.q4_1.bin")
        self.df = pd.DataFrame(data={'x': [i for i in range(10)], 'y': [2 * j + 3 for j in range(10)]})

    '''
    def test_generate_tokens(self):
        prompt = """
        Below is an instruction that describes a task. Write a response that appropriately completes the request in markdown format, using one code block without explanations.
        
        ### Instruction: write a hello world statement in java
        ### Response:
        """
        tokens = self.llm.model.tokenize(prompt)

        reply=""
        for token in self.llm.model.generate(tokens):
            next_token = self.llm.model.detokenize(token)
            print(next_token, end="")
            reply+=next_token

        assert(bool(reply))
    '''
    
    def test_generate_pandas(self):
        df_details = f"""You are provided with a pandas dataframe ddff. ddff has {self.df.shape[0]} rows and {self.df.shape[1]} columns, and the list of its columns are {list(self.df.columns)}.
        Here is a header of ddff:\n{self.df.head().to_string(index=False)}
        
        """
        prompt = f"""
        Below is an instruction that describes a programming task. Write a response in python that appropriately completes the request in markdown format, using one code block without explanations.
        
        ### Instruction: {df_details} Plot a straight line graph of y and x.
        ### Response:
        """
        #print(prompt)

        tokens = self.llm.model.tokenize(prompt)

        reply=""
        for token in self.llm.model.generate(tokens):
            next_token = self.llm.model.detokenize(token)
            # print(next_token, end="")
            reply+=next_token

        assert(bool(reply))
        return reply
    
    def test_code_pandas(self):
        ddff = self.df.copy()
        start_token = r"```(python)\r"
        end_token = r"```"
        reply = self.test_generate_pandas()
        
        back = reply[re.search(start_token, reply, re.M).end(): ]      
        code = back[:re.search(end_token, back).start()]
        print(reply)
        
        try:
            exec(code)
        except Exception as e:
            print(f"{e.__class__.__name__}: {e}")


if __name__ == '__main__':
    unittest.main()