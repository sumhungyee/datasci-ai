import pandas as pd
import matplotlib.pyplot as plt
import warnings
import ast
from datasci_ai.errors import LanguageError, CodeDetectionError, CodeGenerationError, IllegalLoadingError, IllegalCodeError

class AIDataFrame(pd.DataFrame):
    def __init__(self, llm, data=None, index=None, columns=None, dtype=None, copy=None):
        super().__init__(data=data, index=index, columns=columns, dtype=dtype, copy=copy)
        self.llm = llm
        self.name = "tempdf"
        self.df_details = f"""You are provided a pandas dataframe stored in the variable {self.name}. {self.name} has {self.shape[0]} rows and {self.shape[1]} columns, and the here is a dictionary of its columns and their datatypes:{dict(self.dtypes)}.\n"""
        self.prompt = """
{df_details}Below is an instruction that describes a programming task. Write a response that appropriately completes the request and provide code to perform some operations on this existing DataFrame {name}. Write in markdown format and use only one code block. The code must be safe to run. 
### Instruction: {query}
### Response:
"""

    def request(self, query, verbose=True, addon="", max_iters=5):
        full_query = query + "\n" + addon if addon else query
        full_prompt = self.prompt.format(df_details=self.df_details, query=full_query, name=self.name)
        start_token = r"```(python)?\r"
        reply = self.llm.generate_reply(full_prompt, verbose=verbose) # returns a Reply
          
        if max_iters > 0:
            # Try extracting code from reply
            try:
                code = reply.extract_code(start_token=start_token)
            except AttributeError as err:
                language = reply.extract_language()
                if language.lower() != "python" and "python" not in language.lower(): # If language isn't python
                    raise LanguageError(language) from None
                else:
                    warnings.warn(f"Error encountered: {max_iters-1} tries left. Error: {err}")
                    msg = f"You replied with the following response\n{reply.text}\nHowever, the code block was not properly formatted in markdown format."
                    return self.request(query, verbose=verbose, addon=msg, max_iters=max_iters-1)

            except Exception as f:
                warnings.warn(f"Error detecting code! {max_iters-1} tries left.")
                return self.request(query, verbose=verbose, max_iters=max_iters-1)
            
            # Try executing code
            try:
                ai_df = self.safe_exec(code)

            # Capture illegal operations first. Allow model to retry for loading errors, but not illegal code errors eg. unsafe operations.
            except IllegalLoadingError as i:
                warnings.warn(f"Error encountered: {max_iters-1} tries left. Error: {i}")
                msg = f"{self.name} already contains data and there is no need to create a new dataframe or load it."
                plt.close('all')
                return self.request(query, verbose=verbose, addon=msg, max_iters=max_iters-1)
            
            except IllegalCodeError as ic:
                raise ic from None
            
            # Other exceptions can be handled by retrying.
            except Exception as e: 
                warnings.warn(f"Error encountered: {max_iters-1} tries left. Error: {e}")               
                msg = f"You provided this code:\n{code}\nHowever, the following error was thrown:\n{e.__class__.__name__}: {e}\nCorrect these errors, writing code in markdown format using one code block."
                plt.close('all')
                return self.request(query, verbose=verbose, addon=msg, max_iters=max_iters-1)
            
            return AIDataFrame(self.llm, data=ai_df)

            
        else:
            # No more retries left
            try:
                code = reply.extract_code(start_token=start_token)
            except Exception:
                language = reply.extract_language()
                if language.lower() != "python" and "python" not in language.lower(): # If language isn't python
                    raise LanguageError(language) from None
                raise CodeDetectionError() from None
            
            # Try executing code for the last time
            try:
                ai_df = self.safe_exec(code)

            # Raise appropriate illegal errors
            except (IllegalCodeError, IllegalLoadingError) as illegal:
                raise illegal from None

            # Raise all others as code generation errors
            except Exception as ex:
                raise CodeGenerationError(ex) from None
            
            return AIDataFrame(self.llm, data=ai_df)

    def safe_exec(self, code):
        '''
        This method was designed with the aid of ChatGPT
        
        '''
        restricted_functions = {'exec', 'eval', 'open'}
        disallowed_load_functions = {'read_csv', 'read_parquet', \
                                'DataFrame', 'read_orc', 'read_sas', 'read_spss', \
                                'read_sql_table', 'read_sql_query', 'read_sql', 'read_gbq', \
                                'read_stata'}
        restricted_modules = {'os', 'subprocess', '__builtins__', 'tempfile', 'socket', 'shutil'}

        class RestrictionVisitor(ast.NodeVisitor):
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name):
                    function_name = node.func.id
                    if function_name in restricted_functions:
                        raise IllegalCodeError(function_name)
                    elif function_name in disallowed_load_functions:
                        raise IllegalLoadingError()

                elif isinstance(node.func, ast.Attribute):
                    module_name = node.func.value.id if isinstance(node.func.value, ast.Name) else node.func.value.value.id
                    function_name = node.func.attr
                    if module_name in restricted_modules:
                        raise IllegalCodeError(module_name, function=False) 
                    if function_name in restricted_functions:
                        raise IllegalCodeError(function_name)
                    elif function_name in disallowed_load_functions:
                        raise IllegalLoadingError()

                ast.NodeVisitor.generic_visit(self, node)
        

            def visit_Import(self, node):
                for alias in node.names:
                    module_name = alias.name.split('.')[0]
                    if module_name in restricted_modules:
                        raise IllegalCodeError(module_name, function=False)            
                ast.NodeVisitor.generic_visit(self, node)

            def visit_Attribute(self, node):
                if isinstance(node.value, ast.Name):
                    module_name = node.value.id
                    if module_name in restricted_modules:
                        raise IllegalCodeError(module_name, function=False)
                ast.NodeVisitor.generic_visit(self, node)


        # redundant
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise SyntaxError("Invalid syntax.") from None

        visitor = RestrictionVisitor()
        visitor.visit(tree)
        exec(f"{self.name} = self.copy()")
        exec(compile(tree, filename='<string>', mode='exec'))
        return eval(f"{self.name}")


    def copy(self, deep=True):
        df = super().copy(deep=deep)
        return AIDataFrame(self.llm, data=df)
    


    
    
