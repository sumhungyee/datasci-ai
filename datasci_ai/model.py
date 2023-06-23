
from ctransformers import AutoModelForCausalLM
import time
from tqdm import tqdm
import re

class LLM:
    def __init__(self, path=None, model_file='WizardCoder-15B-1.0.ggmlv3.q4_1.bin', link='TheBloke/WizardCoder-15B-1.0-GGML', model_type='starcoder'):
        if not path:
            self.model = AutoModelForCausalLM.from_pretrained(link, model_file=model_file)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(path, model_type=model_type)
        self.last_reply = ""
        
    def generate_reply(self, prompt, verbose=False):
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

class Reply:
    def __int__(self, text, time_elapsed):
        self.text = text
        self.time_elapsed = time_elapsed

    def extract_code(self):
        start_token = r"```(python)?\n"
        end_token = r"```"
        back = self.text[re.search(start_token, self.text).end(): ]
        code = back[:re.search(end_token, back).start()]
        return code

    def __str__(self):
        representation = """
        text: {text}
        time elapsed: {time}
        """
        return representation.format(text=self.text, time=self.time_elapsed)
    