import cachetools
import math
import os
import json
import requests
import sseclient
import urllib
import traceback
import logging

# from aleph_alpha_client import Client as aleph_client, CompletionRequest, Prompt
from datetime import datetime
from dataclasses import dataclass
from typing import Callable, Union
# from .huggingface.hf import HFInference

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@dataclass
class ProviderDetails:
    '''
    Args:
        api_key (str): API key for provider
        version_key (str): version key for provider
    '''
    api_key: str
    version_key: str

@dataclass
class InferenceRequest:
    '''
    Args:
        uuid (str): unique identifier for inference request
        model_name (str): name of model to use
        model_tag (str): tag of model to use
        model_provider (str): provider of model to use
        model_parameters (dict): parameters for model
        prompt (str): prompt to use for inference
    '''
    uuid: str
    model_name: str
    model_tag: str
    model_provider: str
    model_parameters: dict
    prompt: str
    chat: bool

@dataclass
class ProablityDistribution:
    '''
    Args:
        log_prob_sum (float): sum of log probabilities
        simple_prob_sum (float): sum of simple probabilities
        tokens (dict): dictionary of tokens and their probabilities
    '''
    log_prob_sum: float
    simple_prob_sum: float
    tokens: dict

@dataclass
class InferenceResult:
    '''
    Args:
        uuid (str): unique identifier for inference request
        model_name (str): name of model to use
        model_tag (str): tag of model to use
        model_provider (str): provider of model to use
        token (str): token returned by inference
        probability (float): probability of token
        top_n_distribution (ProablityDistribution): top n distribution of tokens
    '''
    uuid: str
    model_name: str
    model_tag: str
    model_provider: str
    token: str
    probability: Union[float, None]
    top_n_distribution: Union[ProablityDistribution, None]

InferenceFunction = Callable[[str, InferenceRequest], None]

class InferenceAnnouncer:
    def __init__(self, sse_topic):
        self.sse_topic = sse_topic
        self.cancel_cache = cachetools.TTLCache(maxsize=1000, ttl=60)

    def __format_message__(self, event: str, infer_result: InferenceResult) -> str:
        logger.debug("formatting message")
        encoded = {
            "message": infer_result.token,
            "modelName": infer_result.model_name,
            "modelTag": infer_result.model_tag,
            "modelProvider": infer_result.model_provider,
        }

        if infer_result.probability is not None:
            encoded["prob"] = round(math.exp(infer_result.probability) * 100, 2) 

        if infer_result.top_n_distribution is not None:
            encoded["topNDistribution"] = {
                "logProbSum": infer_result.top_n_distribution.log_prob_sum,
                "simpleProbSum": infer_result.top_n_distribution.simple_prob_sum,
                "tokens": infer_result.top_n_distribution.tokens
            }

        return json.dumps({"data": encoded, "type": event})
    
    def announce(self, infer_result: InferenceResult, event: str):
        if infer_result.uuid in self.cancel_cache:
            return False

        message = None
        if event == "done":
            message = json.dumps({"data": {}, "type": "done"})
        else:
            message = self.__format_message__(event=event, infer_result=infer_result)

        logger.debug(f"Announcing {event} for uuid: {infer_result.uuid}, message: {message}")
        self.sse_topic.publish(message)

        return True

    def cancel_callback(self, message):
        if message['type'] == 'pmessage':
            data = json.loads(message['data'])
            uuid = data['uuid']
            logger.info(f"Received cancel message for uuid: {uuid}")
            self.cancel_cache[uuid] = True      
   
class InferenceManager:
    def __init__(self, sse_topic, llama_model_path):
        self.announcer = InferenceAnnouncer(sse_topic)

    def __error_handler__(self, inference_fn: InferenceFunction, provider_details: ProviderDetails, inference_request: InferenceRequest):
        logger.info(f"Requesting inference from {inference_request.model_name} on {inference_request.model_provider}")
        infer_result = InferenceResult(
            uuid=inference_request.uuid,
            model_name=inference_request.model_name,
            model_tag=inference_request.model_tag,
            model_provider=inference_request.model_provider,
            token=None,
            probability=None,
            top_n_distribution=None
        )

        if not self.announcer.announce(InferenceResult(
            uuid=inference_request.uuid,
            model_name=inference_request.model_name,
            model_tag=inference_request.model_tag,
            model_provider=inference_request.model_provider,
            token="[INITIALIZING]",
            probability=None,
            top_n_distribution=None
        ), event="status"):
            return

        try:
            inference_fn(provider_details, inference_request)
        except requests.exceptions.RequestException as e:
            logging.error(f"RequestException: {e}")
            infer_result.token = f"[ERROR] No response from {infer_result.model_provider } after sixty seconds"
        except ValueError as e:
            if infer_result.model_provider == "huggingface-local":
                infer_result.token = f"[ERROR] Error parsing response from local inference: {traceback.format_exc()}"
                logger.error(f"Error parsing response from local inference: {traceback.format_exc()}")
            else:
                infer_result.token = f"[ERROR] Error parsing response from API: {e}"
                logger.error(f"Error parsing response from API: {e}")
        except Exception as e:
            infer_result.token = f"[ERROR] {e}"
            logger.error(f"Error: {e}")
        finally:
            if infer_result.token is None:
                infer_result.token = "[COMPLETED]"
            self.announcer.announce(infer_result, event="status")
            logger.info(f"Completed inference for {inference_request.model_name} on {inference_request.model_provider}")
    
    def __openai_chat_generation__(self, provider_details: ProviderDetails, inference_request: InferenceRequest):
        openai.api_key = provider_details.api_key

        current_date = datetime.now().strftime("%Y-%m-%d")

        if inference_request.model_name == "gpt-4":
            system_content = "You are GPT-4, a large language model trained by OpenAI. Answer as concisely as possible"
        else:
            system_content = f"You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible. Knowledge cutoff: 2021-09-01 Current date: {current_date}"

        response = openai.ChatCompletion.create(
             model=inference_request.model_name,
             messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": inference_request.prompt},
            ],
            temperature=inference_request.model_parameters['temperature'],
            max_tokens=inference_request.model_parameters['maximumLength'],
            top_p=inference_request.model_parameters['topP'],
            frequency_penalty=inference_request.model_parameters['frequencyPenalty'],
            presence_penalty=inference_request.model_parameters['presencePenalty'],
            stream=True,
            timeout=60
        )

        tokens = ""
        cancelled = False

        for event in response:
            response = event['choices'][0]
            if response['finish_reason'] == "stop":
                break

            delta = response['delta']

            if "content" not in delta:
                continue

            generated_token = delta["content"]
            tokens += generated_token

            infer_response = InferenceResult(
                uuid=inference_request.uuid,
                model_name=inference_request.model_name,
                model_tag=inference_request.model_tag,
                model_provider=inference_request.model_provider,
                token=generated_token,
                probability=None,
                top_n_distribution=None
             )

            if cancelled: continue

            if not self.announcer.announce(infer_response, event="infer"):
                cancelled = True
                logger.info(f"Cancelled inference for {inference_request.uuid} - {inference_request.model_name}")

    def __llama_cpp_text_generation__(self, provider_details: ProviderDetails, inference_request: InferenceRequest):
        with requests.post("http://127.0.0.1:8080/completion",
                           headers={
                               #"Authorization": f"Bearer {provider_details.api_key}",
                               "Content-Type": "application/json",
                           },
                           data=json.dumps({
                               "prompt": inference_request.prompt,
                               "temperature": float(inference_request.model_parameters['temperature']),
                               "top_p": float(inference_request.model_parameters['topP']),
                               "top_k": int(inference_request.model_parameters['topK']),
                               "stop": inference_request.model_parameters['stopSequences'],
                               "frequency_penalty": float(inference_request.model_parameters['frequencyPenalty']),
                               "presence_penalty": float(inference_request.model_parameters['presencePenalty']),
                               "n_predict": int(inference_request.model_parameters['maximumLength']),
                               "stream": True,
                           }),
                           stream=True
                           ) as response:
            if response.status_code != 200:
                raise Exception(f"Request failed: {response.status_code} {response.reason}")

            cancelled = False
            for token in response.iter_lines():
                token = token.decode('utf-8')
                if len(token) == 0:
                    continue
                token_json = json.loads(token[6:])
                if cancelled: continue

                if not self.announcer.announce(InferenceResult(
                        uuid=inference_request.uuid,
                        model_name=inference_request.model_name,
                        model_tag=inference_request.model_tag,
                        model_provider=inference_request.model_provider,
                        token=token_json['content'],
                        probability=None,  # token_json['likelihood']
                        top_n_distribution=None
                ), event="infer"):
                    cancelled = True
                    logger.info(f"Cancelled inference for {inference_request.uuid} - {inference_request.model_name}")

    def llama_cpp_generation(self, provider_details: ProviderDetails, inference_request: InferenceRequest):
        if inference_request.chat:
            self.__error_handler__(self.__llama_cpp_chat_generation__, provider_details, inference_request)
        else:
            self.__error_handler__(self.__llama_cpp_text_generation__, provider_details, inference_request)

    def __llama_cpp_chat_generation__(self, provider_details: ProviderDetails, inference_request: InferenceRequest):
        current_date = datetime.now().strftime("%Y-%m-%d")
        system_content = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
        context = f"You are avenzaid, a large language model based on open-llama, provided by neomorphic. Answer as concisely as possible. Current date: {current_date}"
        prompt = f"{system_content}\n"
        # prompt += f"ASSISTANT'S RULE: {context}\n"
        prompt += f"USER: {inference_request.prompt}\n"
        with requests.post("http://127.0.0.1:8080/completion",
                           headers={
                               #"Authorization": f"Bearer {provider_details.api_key}",
                               "Content-Type": "application/json",
                           },
                           data=json.dumps({
                               "prompt": prompt,
                               "temperature": float(inference_request.model_parameters['temperature']),
                               "top_p": float(inference_request.model_parameters['topP']),
                               "top_k": int(inference_request.model_parameters['topK']),
                               "stop": inference_request.model_parameters['stopSequences'],
                               "frequency_penalty": float(inference_request.model_parameters['frequencyPenalty']),
                               "presence_penalty": float(inference_request.model_parameters['presencePenalty']),
                               "n_predict": int(inference_request.model_parameters['maximumLength']),
                               "stream": True,
                           }),
                           stream=True
                           ) as response:
            if response.status_code != 200:
                raise Exception(f"Request failed: {response.status_code} {response.reason}")

            cancelled = False
            for token in response.iter_lines():
                token = token.decode('utf-8')
                if len(token) == 0:
                    continue
                token_json = json.loads(token[6:])
                if cancelled: continue

                if not self.announcer.announce(InferenceResult(
                        uuid=inference_request.uuid,
                        model_name=inference_request.model_name,
                        model_tag=inference_request.model_tag,
                        model_provider=inference_request.model_provider,
                        token=token_json['content'],
                        probability=None,  # token_json['likelihood']
                        top_n_distribution=None
                ), event="infer"):
                    cancelled = True
                    logger.info(f"Cancelled inference for {inference_request.uuid} - {inference_request.model_name}")
    
    def get_announcer(self):
        return self.announcer 