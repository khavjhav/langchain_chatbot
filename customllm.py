from typing import Any, List, Mapping, Optional
import requests
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from dotenv import load_dotenv
import time
import os

load_dotenv()
url="https://api.replicate.com/v1/models/mistralai/mixtral-8x7b-instruct-v0.1/predictions"
headers = {
  'Content-Type': 'application/json',
  'Authorization': 'Token '+os.getenv("REPLICATE_TOKEN"),

}
class CustomLLM(LLM):


    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        body = {
                    "input": {
                    "top_k": 50,
                    "top_p": 0.9,
                    "prompt": prompt,
                    "temperature": 0.6,
                    "max_new_tokens": 2048,
                    "prompt_template": "<s>[INST] {prompt} [/INST] ",
                    "presence_penalty": 0,
                    "frequency_penalty": 0
                  }
        }
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")\

        first_chain=requests.post(url, json=body, headers=headers)
        time.sleep(20)
        second_chain=requests.get(first_chain.json()["urls"]["get"],headers=headers)
        output=""
        for word in second_chain.json()["output"]:
            output=output+word
        return output
