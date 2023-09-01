
from ctransformers import AutoModelForCausalLM
import time
from datasci_ai.errors import CodeDetectionError
from tqdm import tqdm
import re

class Reply:
    def __init__(self, text, time_elapsed):
        self.text = text
        self.time_elapsed = time_elapsed

    def extract_code(self, start_token=r"```.*?\r") -> str:
        start_token = start_token
        end_token = r"```"
        back = self.text[re.search(start_token, self.text).end(): ]
        code = back[:re.search(end_token, back).start()]
        return code
    
    def extract_language(self, start_token=r"```.*?\r") -> str:
        start_token = start_token
        try:
            return re.search(start_token, self.text).group()[3:-1]
        except Exception:
            raise CodeDetectionError() from None


    def __str__(self):
        representation = """
        text: {text}
        time elapsed: {time}
        """
        return representation.format(text=self.text, time=self.time_elapsed)

class LLM:
    def __init__(self, path=None, model_file='wizardcoder-python-13b-v1.0.Q5_K_S.gguf', link='TheBloke/WizardCoder-Python-13B-V1.0-GGUF', model_type='llama', llm_model=None, temperature=0.25, gpu_layers=20):
        if llm_model:
            self.model = llm_model
        elif not path:
            self.model = AutoModelForCausalLM.from_pretrained(link, model_file=model_file, temperature=temperature, gpu_layers=gpu_layers, context_length=2048)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(path, model_type=model_type, temperature=temperature, gpu_layers=gpu_layers, context_length=2048)
        self.last_reply = ""
        
    def generate_reply(self, prompt, verbose=False) -> Reply:
        self.clear_last_reply()
        start = time.time()
        tokens = self.model.tokenize(prompt)
        for token in self.model.generate(tokens):
            next_token = self.model.detokenize(token)
            if verbose:
                tqdm.write(next_token, end="")
            self.last_reply += next_token
        end = time.time()
        return Reply(self.last_reply, end - start)
        
    def clear_last_reply(self):
        self.last_reply = ""
