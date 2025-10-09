import base64
import copy
from dataclasses import dataclass, field
import datetime as dt
import importlib
import json
import os
import time
from abc import ABC, abstractmethod
from enum import Enum
from importlib import resources
from typing import (Any, Awaitable, Callable, Optional, OrderedDict, TextIO, Type,
                    TypedDict, TypeVar, Union, cast)

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from llmvm.common.container import Container
from llmvm.common.logging_helpers import setup_logging

logging = setup_logging()


T = TypeVar('T')


class DownloadParams(TypedDict):
    url: str
    goal: str
    search_term: str


class TokenPriceCalculator():
    def __init__(
        self,
        price_file: str = 'model_prices_and_context_window.json',
    ):
        self.price_file = resources.files('llmvm') / price_file
        self.prices = self.__load_prices()
        self.absolute_defaults = {
            'anthropic': {
                'max_input_tokens': 200000,
                'max_output_tokens': 4096,
            },
            'openai': {
                'max_input_tokens': 128000,
                'max_output_tokens': 4096,
            },
            'gemini': {
                'max_input_tokens': 2000000,
                'max_output_tokens': 4096,
            },
            'deepseek': {
                'max_input_tokens': 64000,
                'max_output_tokens': 4096,
            },
            'bedrock': {
                'max_input_tokens': 300000,
                'max_output_tokens': 4096,
            },
        }

    def __load_prices(self):
        with open(self.price_file, 'r') as f:  # type: ignore
            json_prices = json.load(f)
            return json_prices

    def __absolute_default(self, executor: Optional[str], key: str) -> int | None:
        if not executor:
            return None

        executor_value = None
        if executor:
            executor_defaults = self.absolute_defaults.get(executor)
            if executor_defaults:
                executor_value = executor_defaults.get(key)
                if executor_value: return executor_value

    def get(self, model: str, key: str, executor: Optional[str] = None) -> Optional[Any]:
        if model in self.prices and key in self.prices[model]:
            return self.prices[model][key]
        elif executor and f'{executor}/{model}' in self.prices and key in self.prices[f'{executor}/{model}']:
            return self.prices[f'{executor}/{model}'][key]
        return None

    def input_price(
        self,
        model: str,
        executor: Optional[str] = None
    ) -> float:
        return self.get(model, 'input_cost_per_token', executor) or 0.0

    def output_price(
        self,
        model: str,
        executor: Optional[str] = None
    ) -> float:
        return self.get(model, 'output_cost_per_token', executor) or 0.0

    def max_input_tokens(
        self,
        model: str,
        default: int,
        executor: Optional[str] = None,
    ) -> int:
        max_input_tokens = self.get(model, 'max_input_tokens', executor) or default or self.__absolute_default(executor, 'max_input_tokens')

        if not max_input_tokens:
            raise ValueError(f'max_input_tokens not found for model {model} and executor {executor} and no default provided.')

        return cast(int, max_input_tokens)

    def max_output_tokens(
        self,
        model: str,
        default: int,
        executor: Optional[str] = None,
    ) -> int:
        max_output_tokens = self.get(model, 'max_output_tokens', executor) or default or self.__absolute_default(executor, 'max_output_tokens')

        if not max_output_tokens:
            raise ValueError(f'max_output_tokens not found for model {model} and executor {executor} and no default provided.')

        return cast(int, max_output_tokens)

def bcl(module_or_path):
    def decorator(cls):
        class NewClass(cls):
            def __init__(self, *args, **kwargs):
                try:
                    if '.' in module_or_path:
                        # Treat it as a module name
                        module = importlib.import_module(module_or_path)
                        self.arg_string = getattr(module, 'arg_string', None)
                    else:
                        # Treat it as a file path
                        self.arg_string = resources.files(module_or_path).read_text()
                except (ImportError, AttributeError, FileNotFoundError):
                    self.arg_string = None

                super(NewClass, self).__init__(*args, **kwargs)

            def print_arg_string(self):
                if self.arg_string:
                    print(f"Decorator argument: {self.arg_string}")
                else:
                    print("Decorator argument not found.")

        NewClass.__name__ = cls.__name__
        NewClass.__doc__ = cls.__doc__
        return NewClass
    return decorator


async def awaitable_none(a: 'AstNode') -> None:
    pass


def none(a: 'AstNode') -> None:
    pass


class ContentEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'to_json') and callable(getattr(obj, 'to_json')):
            return obj.to_json()
        return super().default(obj)


class TokenCountCache:
    _instance = None

    def __new__(cls, max_size: int = 500):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.max_size = max_size
            cls._instance.cache = OrderedDict()
        return cls._instance

    def __init__(self, max_size: int = 500):
        # The initialization will only happen once
        # subsequent calls will not modify max_size
        if not hasattr(self, 'cache'):
            self.max_size = max_size
            self.cache = OrderedDict()

    def _generate_key(self, messages: list[dict[str, Any]]) -> str:
        return str(hash(str(messages)))

    def get(self, messages: list[dict[str, Any]]) -> Optional[int]:
        key = self._generate_key(messages)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, messages: list[dict[str, Any]], token_count: int) -> None:
        key = self._generate_key(messages)
        if key in self.cache:
            self.cache.move_to_end(key)
            self.cache[key] = token_count
            return
        self.cache[key] = token_count
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def clear(self) -> None:
        self.cache.clear()

    def __len__(self) -> int:
        return len(self.cache)


class TokenPerf:
    def __init__(
        self,
        name: str,
        executor_name: str,
        model_name: str,
        prompt_len: int = 0,
        enabled: bool = Container.get_config_variable('profiling', 'LLMVM_PROFILING', default=False),
        log_file: str = Container.get_config_variable(
            'profiling_file',
            'LLMVM_PROFILING_FILE',
            default='~/.local/share/llmvm/trace.log'
        ),
        request_id: str = '',
        total_tokens: int = 0,
    ):
        self._name: str = name
        self._executor: str = executor_name
        self._model: str = model_name
        self._start: float = 0.0
        self._stop: float = 0.0
        self._prompt_len: int = prompt_len
        self._completion_len: int = 0
        self._ticks: list[float] = []
        self.enabled = enabled
        self.log_file = log_file
        self.calculator = TokenPriceCalculator()
        self.request_id = request_id
        self.stop_reason = ''
        self.stop_token = ''
        self.total_tokens = total_tokens
        self.object = None

    def start(self):
        if self.enabled:
            self._start = time.perf_counter()

    def stop(self):
        if self.enabled:
            self._stop = time.perf_counter()

        return self.result()

    def reset(self):
        self._ticks = []

    def result(self):
        if self.enabled:
            ttlt = self._stop - self._start
            ttft = self._ticks[0] - self._start if self._ticks else 0
            completion_time = ttlt - ttft
            try:
                s_tok_sec = len(self._ticks) / ttlt
            except ZeroDivisionError:
                s_tok_sec = 0.0
            try:
                p_tok_sec = self._prompt_len / ttft
            except ZeroDivisionError:
                p_tok_sec = 0.0
            return {
                'name': self._name,
                'executor': self._executor,
                'model': self._model,
                'ttlt': ttlt,
                'ttft': ttft,
                'completion_time': completion_time,
                'prompt_len': self._prompt_len,
                'completion_len': self._completion_len if self._completion_len > 0 else len(self._ticks),
                's_tok_sec': s_tok_sec,
                'p_tok_sec': p_tok_sec,
                'p_cost': self._prompt_len * self.calculator.input_price(self._model, self._executor),
                's_cost': len(self._ticks) * self.calculator.output_price(self._model, self._executor),
                'request_id': self.request_id,
                'stop_reason': self.stop_reason,
                'stop_token': self.stop_token,
                'total_tokens': self.total_tokens,
                'ticks': self.ticks()
            }
        else:
            return {}

    def tick(self):
        if self.enabled:
            self._ticks.append(time.perf_counter())

    def ticks(self):
        if self.enabled:
            return [self._ticks[i] - self._ticks[i - 1] for i in range(1, len(self._ticks))]
        else:
            return []

    def __str__(self):
        if self.enabled:
            res = self.result()
            result = f'{dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")},{res["name"]},{res["executor"]},{res["model"]},{res["ttlt"]},{res["ttft"]},{res["completion_time"]},{res["prompt_len"]},{res["completion_len"]},{res["p_tok_sec"]},{res["s_tok_sec"]},{res["request_id"]},{res["stop_reason"]},{res["stop_token"]},{res["total_tokens"]},{",".join([f"{t:.8f}" for t in res["ticks"]])}'
            return result
        else:
            return ''

    def debug(self):
        if self.enabled:
            res = self.result()
            # output \n to the debug stream without using logging.debug
            import sys
            sys.stderr.write('\n')
            logging.debug(f"ttft: {res['ttft']:.2f} ttlt: {res['ttlt']:.2f} completion_time: {res['completion_time']:.2f}")
            logging.debug(f"prompt_len: {res['prompt_len']} completion_len: {res['completion_len']} model: {res['model']}")
            logging.debug(f"p_tok_sec: {res['p_tok_sec']:.2f} s_tok_sec: {res['s_tok_sec']:.2f} stop_reason: {res['stop_reason']}")
            logging.debug(f"p_cost: ${res['p_cost']:.5f} s_cost: ${res['s_cost']:.5f} request_id: {res['request_id']}")

    def log(self):
        if self.enabled:
            self.debug()
            if not os.path.exists(os.path.expanduser(self.log_file)):
                with open(os.path.expanduser(self.log_file), 'w') as f:
                    f.write('name,executor,model,ttlt,ttft,prompt_tokens,completion_time,prompt_len,completion_len,p_tok_sec,s_tok_sec,p_cost,s_cost,request_id,stop_reason,stop_token,total_tokens,ticks\n')
            with open(os.path.expanduser(self.log_file), 'a') as f:
                result = str(self)
                f.write(result + '\n')
                return self.result()
        else:
            return {
                'name': self._name,
                'executor': self._executor,
                'ttlt': 0.0,
                'ttft': 0.0,
                'completion_time': 0.0,
                'prompt_len': 0,
                'completion_len': 0,
                'p_tok_sec': 0.0,
                's_tok_sec': 0.0,
                'p_cost': 0.0,
                's_cost': 0.0,
                'request_id': '',
                'stop_reason': '',
                'stop_token': '',
                'total_tokens': 0,
                'ticks': []
            }


########################################################################################
## Model classes
########################################################################################
class Visitor(ABC):
    @abstractmethod
    def visit(self, node: 'AstNode') -> 'AstNode':
        pass


class AstNode(ABC):
    def __init__(
        self
    ):
        pass

    def accept(self, visitor: Visitor) -> 'AstNode':
        return visitor.visit(self)


class Content(AstNode):
    def __init__(
        self,
        sequence: str | bytes | list['Content'],
        content_type: str = '',
        url: str = '',
    ):
        self.sequence = sequence
        self.url = url
        self.content_type = content_type
        self.original_sequence: object = None

    def __repr__(self):
        return f'Content({self.sequence.__repr__()})'

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def get_str(self) -> str:
        pass

    def to_json(self) -> dict:
        sequence_json = ''
        if isinstance(self.sequence, bytes):
            sequence_json = base64.b64encode(self.sequence).decode('utf-8')
        elif isinstance(self.sequence, str):
            sequence_json = self.sequence
        elif isinstance(self.sequence, list):
            sequence_json = [c.to_json() for c in self.sequence]

        return {
            'type': self.__class__.__name__,
            'sequence': sequence_json,
            'content_type': self.content_type,
            'url': self.url,
            'original_sequence': self.original_sequence,
        }

    @classmethod
    def from_json(cls, data: dict) -> 'Content':
        content_type = data.get('content_type', '')
        sequence = data['sequence']
        url = data.get('url', '')

        if content_type == 'text':
            return TextContent(sequence, url)
        elif content_type == 'image':
            image_type = cast(str, data.get('image_type', 'image/png'))
            return ImageContent(
                base64.b64decode(sequence),
                url,
                image_type,
            )
        elif content_type == 'pdf':
            return PdfContent(
                base64.b64decode(sequence),
                url
            )
        elif content_type == 'file':
            return FileContent(
                base64.b64decode(sequence),
                url,
            )
        # container types
        elif content_type == 'browser':
            # Handle BrowserContent's list of Content
            sequence_contents = [Content.from_json(content_data) for content_data in sequence]
            return BrowserContent(sequence_contents, url)
        elif content_type == 'markdown':
            # Handle MarkdownContent's list of Content
            sequence_contents = [Content.from_json(content_data) for content_data in sequence]
            return MarkdownContent(sequence_contents, url)
        elif content_type == 'approval_request':
            return ApprovalRequest.from_json(data)
        else:
            raise ValueError(f"Unknown content type: {content_type}")


class SupportedMessageContent(Content):
    pass

class BinaryContent(Content):
    def __init__(
        self,
        sequence: bytes,
        content_type: str = '',
        url: str = '',
    ):
        if not isinstance(sequence, bytes):
            raise ValueError('sequence must be a bytes object')

        super().__init__(sequence, content_type, url)

    @abstractmethod
    def get_str(self) -> str:
        pass

    @abstractmethod
    def get_bytes(self) -> bytes:
        pass


class TextContent(SupportedMessageContent):
    def __init__(
        self,
        sequence: str,
        url: str = '',
    ):
        if not isinstance(sequence, str):
            raise ValueError('sequence must be a string')

        super().__init__(sequence, 'text', url)
        self.sequence = sequence

    def __str__(self):
        return self.get_str()

    def __repr__(self):
        return f'TextContent({self.sequence})'

    def get_str(self) -> str:
        return self.sequence


class ImageContent(BinaryContent, SupportedMessageContent):
    def __init__(
        self,
        sequence: bytes,
        url: str = '',
        image_type: str = '',
    ):
        super().__init__(sequence, 'image', url)
        self.sequence = sequence
        self.image_type = image_type

    def __str__(self):
        representation = self.url if self.url else f'{len(self.sequence)} bytes'
        return f'ImageContent({representation})'

    def __repr__(self):
        representation = self.url if self.url else f'{len(self.sequence)} bytes'
        return f'ImageContent({representation})'

    def get_str(self) -> str:
        return self.__str__()

    def get_bytes(self) -> bytes:
        return self.sequence


class PdfContent(BinaryContent):
    def __init__(
        self,
        sequence: bytes,
        url: str = '',
    ):

        super().__init__(sequence, 'pdf', url)
        self.sequence = sequence

    def __str__(self):
        return f'PdfContent({self.url})'

    def is_local(self):
        return os.path.isfile(self.url)

    def get_str(self) -> str:
        logging.debug('PdfContent.get_str() called, [PdfContent] string returned')
        return self.__str__()

    def get_bytes(self) -> bytes:
        return self.sequence


class FileContent(BinaryContent):
    def __init__(
        self,
        sequence: bytes,
        url: str = '',
    ):
        super().__init__(sequence, 'file', url)
        self.sequence = sequence

    def __str__(self):
        return f'FileContent({self.url.__str__()} is_local: {self.is_local()})'

    def __repr__(self):
        return f'FileContent({self.url.__str__()} is_local: {self.is_local()})'

    def is_local(self):
        return os.path.isfile(self.url)

    def get_str(self) -> str:
        if self.is_local():
            with open(self.url, 'r') as f:
                return f.read()
                # return f"<file url={self.url}>\n{f.read()}\n</file>"
        elif isinstance(self.sequence, bytes):
            # convert the bytes to a string and return
            return self.sequence.decode('utf-8')
        else:
            raise ValueError('FileContent.get_str() called on non-local file')

    def get_bytes(self) -> bytes:
        return super().get_bytes()


class ContainerContent(Content):
    def __init__(
        self,
        sequence: list[Content],
        content_type: str,
        url: str = '',
    ):
        if not isinstance(sequence, list):
            raise ValueError('sequence must be a list of Content objects')

        super().__init__(sequence, content_type, url)

    def to_json(self) -> dict:
        return {
            "content_type": self.content_type,
            "url": self.url,
            "sequence": [cast(Content, content).to_json() for content in self.sequence]
        }

class BrowserContent(ContainerContent):
    def __init__(
        self,
        sequence: list[Content],
        url: str = '',
    ):
        # browser sequence usually ImageContent, MarkdownContent
        super().__init__(sequence, 'browser', url)
        self.sequence = sequence

    def __str__(self):
        return f'BrowserContent({self.url}) {self.sequence}'

    def __repr__(self):
        return f'BrowserContent({self.url})'

    def get_str(self):
        return '\n'.join([c.get_str() for c in self.sequence])


class MarkdownContent(ContainerContent):
    def __init__(
        self,
        sequence: list[Content],
        url: str = '',
    ):
        if len(sequence) > 2:
            raise ValueError('MarkdownContent sequence must be a list of length 2')
        super().__init__(sequence, 'markdown', url)
        self.sequence = sequence

    def __str__(self):
        return f'MarkdownContent({self.url})'

    def __repr__(self):
        return f'MarkdownContent({self.url.__str__()} sequence: {self.sequence})'

    def get_str(self) -> str:
        return '\n'.join([c.get_str() for c in self.sequence])


class HTMLContent(ContainerContent):
    def __init__(
        self,
        sequence: list[Content] | str,
        url: str = '',
    ):
        if isinstance(sequence, str):
            sequence = [TextContent(sequence)]
        super().__init__(sequence, 'html', url)
        self.sequence = sequence

    def __str__(self):
        return f'HTMLContent({self.url})'

    def __repr__(self):
        return f'HTMLContent({self.url.__str__()} sequence: {self.sequence})'

    def get_str(self) -> str:
        return '\n'.join([c.get_str() for c in self.sequence])


@dataclass
class SearchResult(TextContent):
    def __init__(self, url: str, title: str, snippet: str, engine: str):
        self.url = url
        self.title = title
        self.snippet = snippet
        self.engine = engine
        super().__init__(url=url, sequence=f"SearchResult(url={self.url}, title={self.title}, snippet={self.snippet}, engine={self.engine})")

    def get_str(self) -> str:
        return f"SearchResult(url={self.url}, title={self.title}, snippet={self.snippet}, engine={self.engine})"

    def __str__(self):
        return self.get_str()

    def __repr__(self):
        return self.get_str()

    def to_json(self) -> dict:
        json_result = super().to_json()
        json_result['url'] = self.url
        json_result['title'] = self.title
        json_result['snippet'] = self.snippet
        json_result['engine'] = self.engine
        return json_result


@dataclass
class YelpResult(TextContent):
    def __init__(self, title: str, link: str, neighborhood: str, snippet: str, reviews: str):
        self.title = title
        self.link = link
        self.neighborhood = neighborhood
        self.snippet = snippet
        self.reviews = reviews
        super().__init__(url=link, sequence=f"YelpResult(title={self.title}, link={self.link}, neighborhood={self.neighborhood}, snippet={self.snippet}, reviews={self.reviews})")

    def get_str(self) -> str:
        return f"YelpResult(title={self.title}, link={self.link}, neighborhood={self.neighborhood}, snippet={self.snippet} reviews={self.reviews[0:100]})"

    def __str__(self):
        return self.get_str()

    def __repr__(self):
        return self.get_str()

    def to_json(self) -> dict:
        json_result = super().to_json()
        json_result['title'] = self.title
        json_result['link'] = self.link
        json_result['neighborhood'] = self.neighborhood
        json_result['snippet'] = self.snippet
        json_result['reviews'] = self.reviews
        return json_result


@dataclass
class HackerNewsResult(TextContent):
    def __init__(self, title: str, url: str, author: str, comment_text: str, created_at: str):
        self.title = title
        self.url = url
        self.author = author
        self.comment_text = comment_text
        self.created_at = created_at
        super().__init__(url=url, sequence=f"HackerNewsResult(title={self.title}, url={self.url}, author={self.author}, comment_text={self.comment_text}, created_at={self.created_at})")

    def get_str(self) -> str:
        return f"HackerNewsResult(title={self.title}, url={self.url}, author={self.author}, comment_text={self.comment_text}, created_at={self.created_at})"

    def __str__(self):
        return self.get_str()

    def to_json(self) -> dict:
        json_result = super().to_json()
        json_result['title'] = self.title
        json_result['url'] = self.url
        json_result['author'] = self.author
        json_result['comment_text'] = self.comment_text
        return json_result

# ApprovalRequest class - Temporarily disabled, will be re-enabled in a future update
# Keeping as stub to prevent import errors
class ApprovalRequest(TextContent):
    """Stub class for ApprovalRequest - approval functionality temporarily disabled"""
    def __init__(
        self,
        command: str,
        working_directory: str = '',
        justification: str = '',
        session_id: str = '',
    ):
        # Just create as TextContent for now
        super().__init__(f"Command: {command}", url='')
        self.command = command
        self.working_directory = working_directory
        self.justification = justification
        self.session_id = session_id
        self.execution_id = ""
        self.content_type = "text"  # Use text instead of approval_request

    def get_str(self) -> str:
        return f"Command: {self.command}"

    def __str__(self):
        return self.get_str()

    def to_json(self) -> dict:
        return super().to_json()

    @classmethod
    def from_json(cls, data: dict) -> 'ApprovalRequest':
        return cls(
            command=data.get('command', ''),
            working_directory=data.get('working_directory', ''),
            justification=data.get('justification', ''),
            session_id=data.get('session_id', ''),
        )


class Message(AstNode):
    def __init__(
        self,
        message: list[Content],
        hidden: bool = False,
    ):
        if not isinstance(message, list):
            raise ValueError('message must be a list of Content objects')

        self.message: list[Content] = message
        self.pinned: int = 0  # 0 is not pinned, -1 is pinned last, anything else is pinned
        self.prompt_cached: bool = False
        self.hidden: bool = hidden

    @abstractmethod
    def role(self) -> str:
        pass

    @abstractmethod
    def get_str(self) -> str:
        pass

    def to_json(self) -> dict:
        return {
            "role": self.role(),
            "message": [cast(Content, content).to_json() for content in self.message],
            "pinned": self.pinned,
            "prompt_cached": self.prompt_cached,
            "hidden": self.hidden,
        }

    @classmethod
    def from_json(cls, data: dict) -> 'Message':
        role = data.get('role')
        messages = [Content.from_json(content_data) for content_data in data.get('message', [])]
        prompt_cached = data.get('prompt_cached', False)
        hidden = data.get('hidden', False)
        pinned = data.get('pinned', 0)
        if role == 'user':
            user = User(messages, hidden)
            user.prompt_cached = prompt_cached
            user.pinned = pinned
            return user
        elif role == 'system':
            system = System(messages[0].get_str())
            system.prompt_cached = prompt_cached
            system.pinned = pinned
            return system
        elif role == 'assistant':
            assistant = Assistant(messages[0], hidden)
            assistant.prompt_cached = prompt_cached
            assistant.pinned = pinned
            return assistant
        else:
            raise ValueError(f'Role type not supported {role}, from {data}')

class User(Message):
    def __init__(
        self,
        message: Content | list[Content],
        hidden: bool = False,
    ):
        if not isinstance(message, list):
            message = [message]

        # check to see if all elements are Content
        if not all(isinstance(m, Content) for m in message):
            message_types = ', '.join([str(type(m)) for m in message])
            raise ValueError('User message must be a Content object or list of Content objects, got: ' + message_types)

        super().__init__(message, hidden)

    def role(self) -> str:
        return 'user'

    def __str__(self):
        return self.get_str()

    def get_str(self):
        def content_str(content) -> str:
            if isinstance(content, Content):
                return content.get_str()
            elif isinstance(content, list):
                return '\n'.join([content_str(c) for c in content])
            elif isinstance(content, AstNode):
                return str(content)
            else:
                raise ValueError(f'Unsupported content type for User.get_str(): {type(content)}')

        if isinstance(self.message, Content):
            return self.message.get_str()

        return '\n'.join([content_str(c) for c in self.message])

    def __repr__(self):
        return f'User({self.message.__repr__()})'

    def __add__(self, other):
        a, b = coerce_types(str(self), other)
        return a + b  # type: ignore

    def __radd__(self, other):
        a, b = coerce_types(other, str(self))
        return a + b  # type: ignore


class Developer(Message):
    def __init__(
        self,
        message: str = '''
            You are a helpful assistant.
        '''
    ):
        if not isinstance(message, str):
            raise ValueError('Developer message must be a string')

        super().__init__([TextContent(message)])

    def role(self) -> str:
        return 'developer'

    def __str__(self):
        return self.get_str()

    def __repr__(self):
        return f'Developer({self.message.__repr__()})'

    def get_str(self) -> str:
        return self.message[0].get_str()


class System(Message):
    def __init__(
        self,
        message: str = '''
            You are a helpful assistant.
            Dont make assumptions about what values to plug into functions.
            Ask for clarification if a user request is ambiguous.
        '''
    ):
        if not isinstance(message, str):
            raise ValueError('System message must be a string')

        super().__init__([TextContent(message)])

    def role(self) -> str:
        return 'system'

    def __str__(self):
        return self.get_str()

    def __repr__(self):
        return f'System({self.message.__repr__()})'

    def get_str(self) -> str:
        return self.message[0].get_str()


class Assistant(Message):
    def __init__(
        self,
        message: Content | list[Content],
        thinking: Optional[Content] = None,
        error: bool = False,
        system_context: object = None,
        llm_call_context: object = None,
        stop_reason: str = '',
        stop_token: str = '',
        perf_trace: object = None,
        hidden: bool = False,
        total_tokens: int = 0,
        underlying: object = None,
        response_id: Optional[str] = None,
        round_id: Optional[str] = None,
        inference_number: Optional[int] = None,
    ):
        if isinstance(message, list):
            super().__init__(message, hidden)
        else:
            super().__init__([message], hidden)
        self.error = error
        self.thinking = thinking
        self._system_context = system_context,
        self._llm_call_context: object = llm_call_context
        self.stop_reason: str = stop_reason
        self.stop_token: str = stop_token
        self.perf_trace: object = perf_trace
        self.total_tokens: int = total_tokens
        self.underlying = underlying
        self.response_id: Optional[str] = response_id  # OpenAI Response API response ID for conversation continuation
        self.round_id: Optional[str] = round_id  # Track which round this message is from
        self.inference_number: Optional[int] = inference_number  # Track which inference within the round

    def role(self) -> str:
        return 'assistant'

    def __str__(self):
        return self.get_str()

    def get_str(self):
        return ' '.join([str(m.get_str()) for m in self.message])

    def __add__(self, other):
        def str_str(x):
            if hasattr(x, 'get_str'):
                return x.get_str()
            return str(x)

        assistant = Assistant(
            message=TextContent(str_str(self.message) + str_str(other)),
            system_context=self._system_context,
            llm_call_context=self._llm_call_context,
            stop_reason=self.stop_reason,
            stop_token=self.stop_token,
        )
        return assistant

    def __repr__(self):
        if self.error:
            return f'Assistant({self.message[0].__repr__()} {self.error})'
        else:
            return f'Assistant({self.message[0].__repr__()})'

    def to_json(self):
        json_result = super().to_json()
        json_result['error'] = self.error
        json_result['system_context'] = self._system_context
        json_result['stop_reason'] = self.stop_reason
        json_result['stop_token'] = self.stop_token
        json_result['total_tokens'] = self.total_tokens
        return json_result

    @classmethod
    def from_json(cls, data: dict) -> 'Assistant':
        assistant = cast(Assistant, super().from_json(data))
        assistant.error = data.get('error')
        assistant._system_context = data.get('system_context')
        assistant.stop_reason = cast(str, data.get('stop_reason'))
        assistant.stop_token = cast(str, data.get('stop_token'))
        assistant.total_tokens = cast(int, data.get('total_tokens'))
        return assistant


ContentContent = Union[str, Content, User, Assistant]

########################################################################################
## Interface classes
########################################################################################
class Executor(ABC):
    def __init__(
        self,
        default_model: str,
        api_endpoint: str,
        api_key: str,
        default_max_input_len: int,
        default_max_output_len: int,
    ):
        self._default_model = default_model
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.default_max_input_len = default_max_input_len
        self.default_max_output_len = default_max_output_len

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    async def aexecute(
        self,
        messages: list['Message'],
        max_output_tokens: int = 4096,
        temperature: float = 0.0,
        stop_tokens: list[str] = [],
        model: Optional[str] = None,
        thinking: int = 0,
        stream_handler: Optional[Callable[['AstNode'], Awaitable[None]]] = None,
    ) -> 'Assistant':
        pass

    @property
    def default_model(
        self,
    ) -> str:
        return self._default_model

    @default_model.setter
    def default_model(
        self,
        default_model: str,
    ) -> None:
        self._default_model = default_model

    def max_input_tokens(
        self,
        model: Optional[str] = None,
    ) -> int:
        if model: return TokenPriceCalculator().max_input_tokens(model=model, default=self.default_max_input_len, executor=self.name())
        else: return self.default_max_input_len

    def max_output_tokens(
        self,
        model: Optional[str] = None,
    ) -> int:
        if model: return TokenPriceCalculator().max_output_tokens(model=model, default=self.default_max_output_len, executor=self.name())
        else: return self.default_max_output_len

    @abstractmethod
    def execute(
        self,
        messages: list['Message'],
        max_output_tokens: int = 2048,
        temperature: float = 1.0,
        stop_tokens: list[str] = [],
        model: Optional[str] = None,
        stream_handler: Optional[Callable[['AstNode'], None]] = None,
    ) -> 'Assistant':
        pass

    @abstractmethod
    def to_dict(self, message: 'Message') -> dict:
        pass

    @abstractmethod
    def from_dict(self, message: dict) -> 'Message':
        pass

    @abstractmethod
    async def count_tokens(
        self,
        messages: list['Message'],
    ) -> int:
        pass

    @abstractmethod
    async def count_tokens_dict(
        self,
        messages: list[dict[str, Any]],
    ) -> int:
        pass

    @abstractmethod
    def user_token(
        self
    ) -> str:
        pass

    @abstractmethod
    def assistant_token(
        self
    ) -> str:
        pass

    @abstractmethod
    def append_token(
        self
    ) -> str:
        pass

    @abstractmethod
    def scratchpad_token(
        self
    ) -> str:
        pass

    @abstractmethod
    def unpack_and_wrap_messages(self, messages: list[Message], model: Optional[str] = None) -> list[dict[str, str]]:
        pass

    def does_not_stop(self, model: Optional[str]) -> bool:
        if not model: model = self.default_model
        if 'o1' in model or 'o3' in model or 'o4' in model or 'Llama' in model or 'grok' in model or 'gpt-5' in model:
            return True
        else:
            return False

def coerce_types(a, b):
    # Function to check if a string can be converted to an integer or a float
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def is_float(x):
        return np.isscalar(x) and isinstance(x, (float, np.floating))

    def is_aware(dt):
        return dt.tzinfo is not None and dt.tzinfo.utcoffset(dt) is not None

    if isinstance(a, FunctionCallMeta):
        a = a.result()

    if isinstance(b, FunctionCallMeta):
        b = b.result()

    if isinstance(a, Assistant):
        a = a.get_str()

    if isinstance(b, Assistant):
        b = b.get_str()

    # If either operand is a string and represents a number, convert it
    if isinstance(a, str) and is_number(a):
        a = int(a) if '.' not in a else float(a)
    if isinstance(b, str) and is_number(b):
        b = int(b) if '.' not in b else float(b)

    if isinstance(a, dt.date):
        a = dt.datetime(a.year, a.month, a.day)

    if isinstance(b, dt.date):
        b = dt.datetime(b.year, b.month, b.day)

    if isinstance(a, dt.datetime) and isinstance(b, dt.datetime):
        if is_aware(a) and is_aware(b):
            return a, b
        elif not is_aware(a) and not is_aware(b):
            return a, b
        else:
            a = a.replace(tzinfo=None)
            b = b.replace(tzinfo=None)

    # If either operand is a string now, convert both to strings
    if isinstance(a, str) or isinstance(b, str):
        return str(a), str(b)

    # If they are of the same type, return them as-is
    if type(a) is type(b):
        return a, b

    # numpy and python floats
    if is_float(a) and is_float(b):
        return float(a), float(b)  # type: ignore

    # If one is a float and the other an int, convert the int to float
    if isinstance(a, float) and isinstance(b, int):
        return a, float(b)
    if isinstance(b, float) and isinstance(a, int):
        return float(a), b

    if isinstance(a, dt.datetime) and isinstance(b, dt.timedelta):
        return a, b
    if isinstance(b, dt.datetime) and isinstance(a, dt.timedelta):
        return a, b

    raise TypeError(f"Cannot coerce types {type(a)} and {type(b)} to a common type")

def coerce_to(a: Any, type_var: Type[T]) -> Any:
    # Helper functions
    def is_number(s: str) -> bool:
        try:
            float(s)
            return True
        except ValueError:
            return False

    if isinstance(a, type_var):
        return a

    if isinstance(a, FunctionCallMeta):
        return coerce_to(a.result(), type_var)
    if isinstance(a, User):
        a = a.get_str()
    if isinstance(a, Assistant):
        a = a.get_str()
    if isinstance(a, TextContent):
        a = a.get_str()

    if isinstance(a, str):
        if type_var == bool:
            return a.lower() in ('true', 'yes', '1', 'on')
        elif type_var in (int, float) and is_number(a):
            return type_var(a)
        elif type_var == dt.datetime:
            try:
                return dt.datetime.fromisoformat(a)
            except ValueError:
                pass  # If it fails, we'll raise TypeError at the end

    if isinstance(a, dt.date) and type_var == dt.datetime:
        return dt.datetime(a.year, a.month, a.day)

    if isinstance(a, dt.datetime) and type_var == dt.date:
        return dt.datetime(a.year, a.month, a.day)

    if type_var == str:
        if isinstance(a, dt.datetime):
            return a.isoformat()
        elif isinstance(a, dt.date):
            return a.isoformat()
        elif isinstance(a, list):
            return ' '.join([str(n) for n in a])
        elif isinstance(a, dict):
            return ' '.join([f'{k}: {v}' for k, v in a.items()])
        return str(a)

    if type_var in (int, float, np.floating, np.number):
        if isinstance(a, (int, float, np.number, np.floating)):
            return type_var(a)

    if type_var == bool:
        if isinstance(a, (int, float, np.number, bool)):
            return bool(a)

    raise TypeError(f"Cannot coerce type {type(a)} with value {str(a)[0:50]} to {type_var}")

class TokenCompressionMethod(Enum):
    AUTO = 0
    LIFO = 1
    SIMILARITY = 2
    MAP_REDUCE = 3
    SUMMARY = 4

    @staticmethod
    def from_str(input_str: str) -> 'TokenCompressionMethod':
        if not input_str:
            return TokenCompressionMethod.AUTO
        normalized_str = input_str.upper().replace('MAPREDUCE', 'MAP_REDUCE')
        try:
            return TokenCompressionMethod[normalized_str]
        except KeyError:
            raise ValueError(f"Unknown token compression method: {input_str}")

    @staticmethod
    def get_str(method: 'TokenCompressionMethod') -> str:
        return method.name.lower().replace('_', ' ')

class LLMCall():
    def __init__(
        self,
        user_message: 'Message',
        context_messages: list['Message'],
        executor: Executor,
        model: str,
        temperature: float,
        max_prompt_len: int,
        completion_tokens_len: int,
        prompt_name: str,
        stop_tokens: list[str] = [],
        thinking: int = 0,
        stream_handler: Callable[['AstNode'], Awaitable[None]] = awaitable_none,
        previous_response_id: Optional[str] = None
    ):
        self.user_message = user_message
        self.context_messages = context_messages
        self.executor = executor
        self.model = model
        self.temperature = temperature
        self.max_prompt_len = max_prompt_len
        self.completion_tokens_len = completion_tokens_len
        self.prompt_name = prompt_name
        self.stop_tokens = stop_tokens
        self.thinking = thinking
        self.stream_handler = stream_handler
        self.previous_response_id = previous_response_id

    def copy(self):
        return LLMCall(
            user_message=copy.deepcopy(self.user_message),
            context_messages=copy.deepcopy(self.context_messages),
            executor=self.executor,
            model=self.model,
            temperature=self.temperature,
            max_prompt_len=self.max_prompt_len,
            completion_tokens_len=self.completion_tokens_len,
            prompt_name=self.prompt_name,
            stop_tokens=self.stop_tokens,
            thinking=self.thinking,
            stream_handler=self.stream_handler,
            previous_response_id=self.previous_response_id,
        )


class Controller():
    def __init__(
        self,
    ):
        pass

    @abstractmethod
    def aexecute_llm_call(
        self,
        llm_call: LLMCall,
        query: str,
        original_query: str,
        compression: TokenCompressionMethod = TokenCompressionMethod.AUTO,
    ) -> 'Assistant':
        pass

    @abstractmethod
    def execute_llm_call(
        self,
        llm_call: LLMCall,
        query: str,
        original_query: str,
        compression: TokenCompressionMethod = TokenCompressionMethod.AUTO,
    ) -> 'Assistant':
        pass

    @abstractmethod
    def get_executor() -> Executor:
        pass


class TokenNode(AstNode):
    def __init__(
        self,
        token: str,
    ):
        super().__init__()
        self.token = token

    def __str__(self):
        return self.token

    def __repr__(self):
        return f'TokenNode({self.token})'


class TokenThinkingNode(TokenNode):
    def __init__(
        self,
        token: str,
    ):
        super().__init__(token)

    def __repr__(self):
        return f'TokenThinkingNode({self.token})'


class TokenStopNode(AstNode):
    def __init__(
        self,
        print_str: str = '',
    ):
        super().__init__()
        self.print_str = print_str

    def __str__(self):
        return self.print_str

    def __repr__(self):
        return f'TokenStopNode(print_str={self.print_str!r})'


class StreamingStopNode(AstNode):
    def __init__(
        self,
        print_str: str = '\n',
    ):
        super().__init__()
        self.print_str = print_str

    def __str__(self):
        return self.print_str

    def __repr__(self):
        return f'StreamingStopNode(print_str={self.print_str!r})'


class QueueBreakNode(AstNode):
    def __init__(
        self,
    ):
        super().__init__()

    def __str__(self):
        return '\n'

    def __repr__(self):
        return 'QueueBreakNode()'



class StreamNode(AstNode):
    def __init__(
        self,
        obj: object,
        type: str,
        metadata: object = None,
    ):
        super().__init__()
        self.obj = obj
        self.type = type
        self.metadata = metadata

    def __str__(self):
        return f'StreamNode{str(self.obj)}'

    def __repr__(self):
        return 'StreamNode()'


class DebugNode(AstNode):
    def __init__(
        self,
        debug_str: str,
    ):
        super().__init__()
        self.debug_str = debug_str

    def __str__(self):
        return f'DebugNode({self.debug_str})'

    def __repr__(self):
        return 'DebugNode()'


class InferenceStartNode(AstNode):
    def __init__(
        self,
        model: str,
        prompt_tokens: int = 0,
        request_id: str = "",
        round_id: str = "1",
        depth_level: int = 0,
        prompt_file: str = "",
        attempt_number: int = 1,
    ):
        super().__init__()
        self.model = model
        self.prompt_tokens = prompt_tokens
        self.request_id = request_id
        self.round_id = round_id
        self.depth_level = depth_level
        self.prompt_file = prompt_file
        self.attempt_number = attempt_number
        self.timestamp = time.time()

    def __str__(self):
        prompt_info = f' prompt="{self.prompt_file}"' if self.prompt_file else ''
        return f'<inference-started round="{self.round_id}" depth="{self.depth_level}"{prompt_info}>\n'

    def __repr__(self):
        return f'InferenceStartNode(model={self.model}, round_id={self.round_id}, depth={self.depth_level}, prompt={self.prompt_file})'


class InferenceEndNode(AstNode):
    def __init__(
        self,
        success: bool,
        duration: float,
        total_tokens: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        error: str = "",
        round_id: str = "1",
        depth_level: int = 0,
        attempt_number: int = 1,
        response_id: str = "",
        previous_response_id: str = "",
    ):
        super().__init__()
        self.success = success
        self.duration = duration
        self.total_tokens = total_tokens
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.error = error
        self.round_id = round_id
        self.depth_level = depth_level
        self.attempt_number = attempt_number
        self.response_id = response_id
        self.previous_response_id = previous_response_id
        self.timestamp = time.time()

    def __str__(self):
        if self.success:
            return f'<inference-succeeded round="{self.round_id}" depth="{self.depth_level}">\n'
        else:
            return f'<inference-failed round="{self.round_id}" depth="{self.depth_level}">\n'

    def __repr__(self):
        return f'InferenceEndNode(success={self.success}, round_id={self.round_id}, depth={self.depth_level})'


class HelpersExtractedNode(AstNode):
    def __init__(
        self,
        code_blocks: list[str],
        total_blocks: int,
        round_id: str = "1",
        depth_level: int = 0,
        attempt_number: int = 1,
    ):
        super().__init__()
        self.code_blocks = code_blocks
        self.total_blocks = total_blocks
        self.round_id = round_id
        self.depth_level = depth_level
        self.attempt_number = attempt_number
        self.timestamp = time.time()

    def __str__(self):
        return f'<helpers-extracted count="{self.total_blocks}" round="{self.round_id}" depth="{self.depth_level}">\n'

    def __repr__(self):
        return f'HelpersExtractedNode(total_blocks={self.total_blocks}, round_id={self.round_id}, depth={self.depth_level})'


class HelpersExecutionStartNode(AstNode):
    def __init__(
        self,
        code_block: str,
        block_index: int = 0,
        round_id: str = "1",
        depth_level: int = 0,
        helper_name: str = "",
        helper_args_summary: str = "",
        attempt_number: int = 1,
    ):
        super().__init__()
        self.code_block = code_block
        self.block_index = block_index
        self.round_id = round_id
        self.depth_level = depth_level
        self.attempt_number = attempt_number

        # Extract helper details if not provided
        if not helper_name:
            helper_name, helper_args_summary = self.extract_helper_details(code_block)

        self.helper_name = helper_name
        self.helper_args_summary = helper_args_summary
        self.timestamp = time.time()

    @staticmethod
    def extract_helper_details(code: str) -> tuple[str, str]:
        """
        Extract helper name and argument summary from code block.
        Returns: (helper_name, args_summary)

        Examples:
            "download('http://example.com')" → ("download", "http://example.com")
            "BCL.search(query='python')" → ("BCL.search", "query='python'")
            "llm_call(data, 'extract features')" → ("llm_call", "'extract features'")
            "result = analyze(df, method='linear')" → ("analyze", "df, method='linear'")
            "def get_data(): pass" → ("code", "def get_data(): pass")
        """
        import re

        # Python keywords that shouldn't be considered helpers
        PYTHON_KEYWORDS = {
            'def', 'class', 'if', 'elif', 'else', 'for', 'while', 'try', 'except',
            'finally', 'with', 'return', 'yield', 'import', 'from', 'as', 'pass',
            'break', 'continue', 'raise', 'assert', 'del', 'global', 'nonlocal',
            'lambda', 'and', 'or', 'not', 'in', 'is'
        }

        # Strip whitespace and get first line if multi-line
        code = code.strip()
        first_line = code.split('\n')[0] if '\n' in code else code

        # Pattern 1: Assignment with function call
        # result = function(args)
        match = re.search(r'=\s*(\w+(?:\.\w+)?)\s*\((.*?)\)', first_line)
        if match:
            helper_name = match.group(1)
            # Skip if it's a Python keyword
            if helper_name.lower() not in PYTHON_KEYWORDS:
                args_raw = match.group(2)
                args_summary = args_raw[:50] + "..." if len(args_raw) > 50 else args_raw
                return (helper_name, args_summary)

        # Pattern 2: Direct function call
        # function(args)
        match = re.search(r'^(\w+(?:\.\w+)?)\s*\((.*?)\)', first_line)
        if match:
            helper_name = match.group(1)
            # Skip if it's a Python keyword
            if helper_name.lower() not in PYTHON_KEYWORDS:
                args_raw = match.group(2)
                args_summary = args_raw[:50] + "..." if len(args_raw) > 50 else args_raw
                return (helper_name, args_summary)

        # Pattern 3: llm_call specifically (capture instruction)
        if 'llm_call' in code:
            match = re.search(r'llm_call\([^,]*,\s*["\']([^"\']+)["\']', code)
            if match:
                instruction = match.group(1)[:50]
                return ("llm_call", f'"{instruction}"')

        # Fallback: it's just regular Python code, not a helper call
        code_preview = first_line[:80] + "..." if len(first_line) > 80 else first_line
        return ("code", code_preview)

    def __str__(self):
        return f'<helpers-execution-started helper="{self.helper_name}" round="{self.round_id}" depth="{self.depth_level}">\n'

    def __repr__(self):
        return f'HelpersExecutionStartNode(helper={self.helper_name}, round_id={self.round_id}, depth={self.depth_level})'


class HelpersExecutionEndNode(AstNode):
    def __init__(
        self,
        success: bool,
        result: str = "",
        error: str = "",
        duration: float = 0.0,
        round_id: str = "1",
        depth_level: int = 0,
        attempt_number: int = 1,
    ):
        super().__init__()
        self.success = success
        self.result = result
        self.error = error
        self.duration = duration
        self.round_id = round_id
        self.depth_level = depth_level
        self.attempt_number = attempt_number
        self.timestamp = time.time()

    def __str__(self):
        status = "success" if self.success else "error"
        return f'<helpers-execution-ended status="{status}" duration="{self.duration:.2f}s" round="{self.round_id}" depth="{self.depth_level}">\n'

    def __repr__(self):
        return f'HelpersExecutionEndNode(success={self.success}, round_id={self.round_id}, depth={self.depth_level}, duration={self.duration})'


@dataclass
class ExecutionTraceItem:
    """Single item in execution trace with detailed metadata"""
    round_id: str              # "1", "1.2", "1.2.3"
    depth_level: int           # 0 = main, 1 = nested, etc.
    event_type: str            # "inference_start", "helpers_extracted", "helper_exec_start", etc.

    # Prompt information
    prompt_file: str = ""      # "python_continuation_execution.prompt", "search.prompt", etc.
    model: str = ""            # "gpt-5", "claude-sonnet-4", etc.
    response_id: str = ""      # OpenAI Response API response_id for this inference
    previous_response_id: str = ""  # Previous response_id used to continue conversation

    # Helper execution details
    helper_name: str = ""      # "download", "BCL.search", "llm_call", etc.
    helper_args_summary: str = ""  # Summary of arguments
    code_snippet: str = ""     # First 100 chars of code block

    # Execution results
    blocks_count: int = 0
    success: bool = True
    duration: float = 0.0
    error_message: str = ""    # Store error details

    # Timing
    timestamp: float = 0.0

    def get_indent(self) -> str:
        """Get indentation based on depth and round nesting"""
        # Base indent from depth
        indent = "  " * self.depth_level

        # Additional indent for nested rounds (count dots in round_id)
        round_depth = self.round_id.count('.')
        indent += "  " * round_depth

        return indent


class ExecutionSummaryNode(AstNode):
    """Summary of entire execution with detailed trace tree"""
    def __init__(
        self,
        trace_items: list[ExecutionTraceItem],
        total_rounds: int,
        total_helpers: int,
        total_duration: float,
        max_depth: int,
        main_model: str = "",
    ):
        super().__init__()
        self.trace_items = trace_items
        self.total_rounds = total_rounds
        self.total_helpers = total_helpers
        self.total_duration = total_duration
        self.max_depth = max_depth
        self.main_model = main_model
        self.timestamp = time.time()

    def __str__(self):
        """Generate tree view of execution"""
        return self.generate_tree_view()

    def _get_unique_helpers(self) -> dict[str, int]:
        """Get count of each unique helper executed"""
        helper_counts = {}
        for item in self.trace_items:
            if item.event_type == "helper_exec_start" and item.helper_name:
                helper_counts[item.helper_name] = helper_counts.get(item.helper_name, 0) + 1
        return helper_counts

    def _get_unique_prompts(self) -> set[str]:
        """Get list of unique prompt files used"""
        prompts = set()
        for item in self.trace_items:
            if item.event_type == "inference_start" and item.prompt_file:
                prompts.add(item.prompt_file)
        return prompts

    def generate_tree_view(self) -> str:
        """Generate compact tree view of execution trace"""
        lines = []

        # Compact header
        lines.append(f"\n{'='*70}")
        helper_counts = self._get_unique_helpers()
        helper_str = " ".join([f"{h}({c}x)" for h, c in sorted(helper_counts.items())])
        model_str = f" | {self.main_model}" if self.main_model else ""
        lines.append(f"{self.total_rounds} rounds, {self.total_helpers} helpers, {self.total_duration:.2f}s{model_str} | {helper_str}")
        lines.append(f"{'='*70}\n")

        # Group trace items by round and inference
        current_round = ""
        inference_nums = {}  # Track inference number per round
        inference_data = {}  # {(round, inf): {'helpers': [], 'prev_resp': '', 'resp': ''}}

        for item in self.trace_items:
            if item.event_type == "inference_start":
                current_round = item.round_id
                # Reset inference counter when round changes, otherwise increment
                if current_round not in inference_nums:
                    inference_nums[current_round] = 1
                else:
                    inference_nums[current_round] += 1
                current_inference_num = inference_nums[current_round]
                key = (current_round, current_inference_num)
                if key not in inference_data:
                    inference_data[key] = {
                        'helpers': [],
                        'prev_resp': '',
                        'resp': '',
                        'prompt': item.prompt_file,
                        'model': item.model,
                        'depth': item.depth_level
                    }

            elif item.event_type == "helper_exec_start":
                key = (current_round, current_inference_num)
                if key in inference_data:
                    inference_data[key]['helpers'].append({
                        'name': item.helper_name,
                        'code': item.code_snippet,
                        'success': None,
                        'duration': 0,
                        'error': ''
                    })

            elif item.event_type == "helper_exec_end":
                key = (current_round, current_inference_num)
                if key in inference_data and inference_data[key]['helpers']:
                    inference_data[key]['helpers'][-1]['success'] = item.success
                    inference_data[key]['helpers'][-1]['duration'] = item.duration
                    inference_data[key]['helpers'][-1]['error'] = item.error_message

            elif item.event_type == "inference_end":
                key = (current_round, current_inference_num)
                if key in inference_data:
                    inference_data[key]['prev_resp'] = item.previous_response_id
                    inference_data[key]['resp'] = item.response_id

        # Display rounds compactly
        last_round = ""
        for (round_id, inf_num), data in sorted(inference_data.items()):
            indent = "  " * data['depth']

            # Show round header only when round changes
            if round_id != last_round:
                if last_round:
                    lines.append("")  # Blank line between rounds
                last_round = round_id
                prompt_str = f" [{data['prompt']}]" if data['prompt'] else ""
                model_str = f" ({data['model']})" if data['model'] else ""
                lines.append(f"{indent}[R{round_id}]{prompt_str}{model_str}")

            # Show each helper execution inline
            if data['helpers']:
                for i, helper in enumerate(data['helpers']):
                    is_last = (i == len(data['helpers']) - 1)
                    prefix = "└─" if is_last else "├─"

                    # Determine helper name and format code
                    if helper['name'].lower() == 'code' or not helper['name']:
                        # Show as Code(...) for regular Python code
                        helper_display = f"Code({helper['code'][:80]})"
                    else:
                        # Show helper name with args
                        helper_display = f"{helper['name']}({helper['code'][:80]})"

                    status = "✓" if helper['success'] else "✗"
                    result = f"{indent}{prefix} #{inf_num}: {helper_display} → {status} ({helper['duration']:.2f}s)"

                    if not helper['success'] and helper['error']:
                        error_preview = helper['error'][:60] + "..." if len(helper['error']) > 60 else helper['error']
                        result += f" (Error: {error_preview})"

                    lines.append(result)

                # Show response IDs on last helper line if available
                if data['prev_resp'] or data['resp']:
                    response_parts = []
                    if data['prev_resp']:
                        prev_short = data['prev_resp'][-5:]
                        response_parts.append(f"prev={prev_short}")
                    if data['resp']:
                        resp_short = data['resp'][-5:]
                        response_parts.append(f"resp={resp_short}")
                    resp_info = f" ({', '.join(response_parts)})"
                    # Append to last line
                    lines[-1] += resp_info
            else:
                # No helpers executed in this inference
                lines.append(f"{indent}└─ #{inf_num}: (no helpers)")
                if data['prev_resp'] or data['resp']:
                    response_parts = []
                    if data['prev_resp']:
                        prev_short = data['prev_resp'][-5:]
                        response_parts.append(f"prev={prev_short}")
                    if data['resp']:
                        resp_short = data['resp'][-5:]
                        response_parts.append(f"resp={resp_short}")
                    lines[-1] += f" ({', '.join(response_parts)})"

        lines.append(f"\n{'='*70}\n")
        return "\n".join(lines)

    def __repr__(self):
        return f'ExecutionSummaryNode(rounds={self.total_rounds}, helpers={self.total_helpers}, duration={self.total_duration:.2f}s, model={self.main_model})'


class Statement(AstNode):
    def __init__(
        self,
    ):
        self._result: object = None

    def __str__(self):
        if self._result:
            return str(self._result)
        else:
            return str(type(self))

    def result(self):
        return self._result

    def token(self):
        return 'statement'


class DataFrame(Statement):
    def __init__(
        self,
        elements: list,
    ):
        super().__init__()
        self.elements = elements

    def token(self):
        return 'dataframe'


class Call(Statement):
    def __init__(
        self,
    ):
        super().__init__()


class FunctionCallMeta(Call):
    def __init__(
        self,
        callsite: str,
        func: Callable,
        result: object,
        lineno: Optional[int],
    ):
        self.callsite = callsite
        self.func = func
        self._result = result
        self.lineno = lineno

    def result(self) -> object:
        return self._result

    def token(self):
        return 'functioncallmeta'

    def __enter__(self):
        return self._result.__enter__()  # type: ignore

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._result.__exit__(exc_type, exc_val, exc_tb)  # type: ignore

    def __float__(self):
        return float(self._result)  # type: ignore

    def __getattr__(self, name):
        if self._result is not None:
            return getattr(self._result, name)

        raise AttributeError(f"'self._result isn't set, and {self.__class__.__name__}' object has no attribute '{name}'")

    def __getitem__(self, key):
        if self._result is not None and hasattr(self._result, '__getitem__'):
            return self._result[key]  # type: ignore

        raise AttributeError(f"{type(self._result)} is not subscriptable")

    def __setstate__(self, state):
        # Directly set _data without going through __getattr__
        self._result = state.get('_result')

    def __getstate__(self):
        # Return a dictionary representing the object's state
        return {'_result': self._result}

    def __str__(self):
        return str(self._result)

    def __repr__(self):
        return self._result.__repr__()

    def get_str(self):
        if hasattr(self._result, 'get_str'):
            return self._result.get_str()  # type: ignore
        else:
            return str(self._result)

    def __add__(self, other):
        a, b = coerce_types(self._result, other)
        return a + b  # type: ignore

    def __sub__(self, other):
        a, b = coerce_types(self._result, other)
        return a - b  # type: ignore

    def __mul__(self, other):
        a, b = coerce_types(self._result, other)
        return a * b  # type: ignore

    def __div__(self, other):
        a, b = coerce_types(self._result, other)
        return a / b  # type: ignore

    def __truediv__(self, other):
        a, b = coerce_types(self._result, other)
        return a / b  # type: ignore

    def __rtruediv__(self, other):
        a, b = coerce_types(other, self._result)
        return a / b  # type: ignore

    def __radd__(self, other):
        a, b = coerce_types(other, self._result)
        return a + b  # type: ignore

    def __rsub__(self, other):
        a, b = coerce_types(other, self._result)
        return a - b  # type: ignore

    def __rmul__(self, other):
        a, b = coerce_types(other, self._result)
        return a * b  # type: ignore

    def __rdiv__(self, other):
        a, b = coerce_types(other, self._result)
        return a / b  # type: ignore

    def __gt__(self, other):
        a, b = coerce_types(self._result, other)
        return a > b  # type: ignore

    def __lt__(self, other):
        a, b = coerce_types(self._result, other)
        return a < b  # type: ignore

    def __ge__(self, other):
        a, b = coerce_types(self._result, other)
        return a >= b  # type: ignore

    def __le__(self, other):
        a, b = coerce_types(self._result, other)
        return a <= b  # type: ignore

    def __rgt__(self, other):
        # Note the order in coerce_types is reversed
        a, b = coerce_types(other, self._result)
        return a > b  # type: ignore

    def __rlt__(self, other):
        a, b = coerce_types(other, self._result)
        return a < b  # type: ignore

    def __rge__(self, other):
        a, b = coerce_types(other, self._result)
        return a >= b  # type: ignore

    def __rle__(self, other):
        a, b = coerce_types(other, self._result)
        return a <= b  # type: ignore

    def __eq__(self, other):
        a, b = coerce_types(self._result, other)
        return a == b

    def __ne__(self, other):
        a, b = coerce_types(self._result, other)
        return a != b

    def __bool__(self):
            return bool(self._result)

    def __hash__(self):
        return hash(self._result)

    def __int__(self):
        return int(self._result)          # type: ignore

    def __index__(self):
        return self._result.__index__()   # type: ignore

    def __bytes__(self):
        return bytes(self._result)        # type: ignore

    def __complex__(self):
        return complex(self._result)      # type: ignore

    def __iter__(self):
            return iter(self._result)         # type: ignore

    def __next__(self):
        return next(self._result)         # type: ignore  # works if _result is an iterator

    def __reversed__(self):
        return reversed(self._result)     # type: ignore

    def __contains__(self, item):
        return item in self._result       # type: ignore

    def __call__(self, *args, **kwargs):
        return self._result(*args, **kwargs)  # type: ignore

    async def __aenter__(self):
        return await self._result.__aenter__()    # type: ignore

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return await self._result.__aexit__(exc_type, exc_val, exc_tb)  # type: ignore

    def __abs__(self):
        return abs(self._result)          # type: ignore

    def __neg__(self):
        return -self._result              # type: ignore

    def __pos__(self):
        return +self._result              # type: ignore

    def __invert__(self):
        return ~self._result              # type: ignore

    def __await__(self):
        return self._result.__await__()   # type: ignore

    def __aiter__(self):
        return self._result.__aiter__()   # type: ignore

    async def __anext__(self):
        return await self._result.__anext__()  # type: ignore

    def __floordiv__(self, other):
        a, b = coerce_types(self._result, other)
        return a // b                     # type: ignore

    def __rfloordiv__(self, other):
        a, b = coerce_types(other, self._result)
        return a // b                     # type: ignore

    def __mod__(self, other):
        a, b = coerce_types(self._result, other)
        return a % b                      # type: ignore

    def __rmod__(self, other):
        a, b = coerce_types(other, self._result)
        return a % b                      # type: ignore

    def __pow__(self, other, modulo=None):
        a, b = coerce_types(self._result, other)
        return pow(a, b, modulo) if modulo is not None else pow(a, b)   # type: ignore

    def __rpow__(self, other):
        a, b = coerce_types(other, self._result)
        return pow(a, b)                  # type: ignore

    def __matmul__(self, other):
        a, b = coerce_types(self._result, other)
        return a @ b                      # type: ignore

    def __rmatmul__(self, other):
        a, b = coerce_types(other, self._result)
        return a @ b                      # type: ignore

    def __lshift__(self, other):
        a, b = coerce_types(self._result, other)
        return a << b                     # type: ignore

    def __rlshift__(self, other):
        a, b = coerce_types(other, self._result)
        return a << b                     # type: ignore

    def __rshift__(self, other):
        a, b = coerce_types(self._result, other)
        return a >> b                     # type: ignore

    def __rrshift__(self, other):
        a, b = coerce_types(other, self._result)
        return a >> b                     # type: ignore

    def __and__(self, other):
        a, b = coerce_types(self._result, other)
        return a & b                      # type: ignore

    def __rand__(self, other):
        a, b = coerce_types(other, self._result)
        return a & b                      # type: ignore

    def __or__(self, other):
        a, b = coerce_types(self._result, other)
        return a | b                      # type: ignore

    def __ror__(self, other):
        a, b = coerce_types(other, self._result)
        return a | b                      # type: ignore

    def __xor__(self, other):
        a, b = coerce_types(self._result, other)
        return a ^ b                      # type: ignore

    def __rxor__(self, other):
        a, b = coerce_types(other, self._result)
        return a ^ b                      # type: ignore

    def __setitem__(self, key, value):
        if hasattr(self._result, '__setitem__'):
            self._result[key] = value     # type: ignore
        else:
            raise TypeError(f"{type(self._result).__name__} does not support item assignment")

    def __dir__(self):
        return sorted(set(dir(type(self)) + dir(self.__dict__) + dir(self._result)))

    def __copy__(self):
        import copy
        return copy.copy(self._result)    # type: ignore

    def __deepcopy__(self, memo):
        import copy
        return copy.deepcopy(self._result, memo)  # type: ignore

    def __delitem__(self, key):
        if hasattr(self._result, '__delitem__'):
            del self._result[key]         # type: ignore
        else:
            raise TypeError(f"{type(self._result).__name__} does not support item deletion")

    def __len__(self):
        return len(self._result)  # type: ignore

    def __format__(self, format_spec):
        return format(self._result, format_spec)


class PandasMeta(Call):
    def __init__(
        self,
        expr_str: str,
        pandas_df,
    ):
        self.expr_str = expr_str
        self.df: pd.DataFrame = pandas_df

    def result(self) -> object:
        return self._result

    def token(self):
        return 'pandasmeta'

    def __str__(self):
        str_acc = ''
        if self.df is not None:
            str_acc += f'info()\n'
            str_acc += f'{self.df.info()}\n\n'  # type: ignore
            str_acc += f'describe()\n'
            str_acc += f'{self.df.describe()}\n\n'  # type: ignore
            str_acc += f'head()\n'
            str_acc += f'{self.df.head()}\n\n'  # type: ignore
            str_acc += '\n'
            str_acc += f'call "to_string()" to get the entire DataFrame as a string\n'
            return str_acc
        else:
            return '[]'

    def __gt__(self, other):
        return self.df > other

    def __lt__(self, other):
        return self.df < other

    def __ge__(self, other):
        return self.df >= other

    def __le__(self, other):
        return self.df <= other

    def __rgt__(self, other):
        return self.df > other

    def __rlt__(self, other):
        return self.df < other

    def __rge__(self, other):
        return self.df >= other

    def __rle__(self, other):
        return self.df <= other

    def __add__(self, other):
        return self.df + other

    def __len__(self):
        return len(self.df)

    def __getattr__(self, name):
        if name in self.__dict__:
            return getattr(self.df, name)
        elif object.__getattribute__(self, 'df') is not None:
            return getattr(self.df, name)
        raise AttributeError(f"'self.df isn't set, and {self.__class__.__name__}' object has no attribute '{name}'")

    def __getitem__(self, key):
        return self.df.__getitem__(key)  # type: ignore

    def __setitem__(self, key, value):
        self.df.__setitem__(key, value)

    def __iter__(self):
        return iter(self.df)

    def __format__(self, format_spec):
        return format(self.pandas_df, format_spec)

    def to_string(self):
        return self.df.to_string()


class FunctionCall(Call):
    def __init__(
        self,
        name: str,
        args: list[dict[str, object]],
        types: list[dict[str, object]],
        context: 'Content' = TextContent(''),
        func: Optional[Callable] = None,
    ):
        super().__init__()
        self.name = name
        self.args = args
        self.types = types
        self.context = context
        self._result: Optional[Content] = None
        self.func: Optional[Callable] = func

    def to_code_call(self):
        arguments = []
        for arg in self.args:
            for k, v in arg.items():
                arguments.append(v)

        str_args = ', '.join([str(arg) for arg in arguments])
        return f'{self.name}({str_args})'

    def to_definition(self):
        definitions = []
        for arg in self.types:
            for k, v in arg.items():
                definitions.append(f'{k}: {v}')

        str_args = ', '.join([str(t) for t in definitions])
        return f'{self.name}({str_args})'

    def token(self):
        return 'function_call'

class Answer(Statement):
    def __init__(
        self,
        result: object = None,
        error: object = None,
    ):
        super().__init__()
        self._result = result
        self.error = error

    def __str__(self):
        if not self.error:
            return str(self.result())
        else:
            return str(self.error)

    def get_str(self):
        return str(self)

    def token(self):
        return 'answer'


########################################################################################
## Pydantic classes
########################################################################################
class DownloadItemModel(BaseModel):
    id: int
    url: str


class ContentModel(BaseModel):
    sequence: Union[list[dict], str, bytes]
    content_type: str
    original_sequence: Optional[Union[list[dict], str, bytes]] = None
    url: str

    class Config:
        from_attributes = True

    def to_content(self) -> Content:
        return Content.from_json(data=self.model_dump())

    @classmethod
    def from_content(cls, content: Content) -> 'ContentModel':
        return cls.model_validate(content.to_json())


def _bytes_to_b64(b: bytes) -> str:          # NEW
    result = base64.b64encode(b).decode('ascii')
    return result


class MessageModel(BaseModel):
    role: str
    content: list[ContentModel]
    pinned: int = 0
    prompt_cached: bool = False
    total_tokens: int = 0  # only used on Assistant messages
    underlying: Any = Field(default=None)
    response_id: Optional[str] = None  # only used on Assistant messages for Response API
    round_id: Optional[str] = None  # track which round this message is from
    inference_number: Optional[int] = None  # track which inference within the round

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={
            bytes: _bytes_to_b64,
        },
    )

    def to_message(self) -> Message:
        content_objects = [c.to_content() for c in self.content]

        if self.role == 'user':
            msg = User(content_objects)
        elif self.role == 'system':
            msg = System(content_objects[0].get_str())
        elif self.role == 'assistant':
            msg = Assistant(content_objects[0], total_tokens=self.total_tokens, underlying=self.underlying, response_id=self.response_id)
        else:
            raise ValueError(f"MessageModel.to_message() Unsupported role: {self.role}")

        msg.pinned = self.pinned
        msg.prompt_cached = self.prompt_cached
        return msg

    @classmethod
    def from_message(cls, message: Message) -> 'MessageModel':
        content_models = [ContentModel.from_content(content) for content in message.message]

        return cls(
            role=message.role(),
            content=content_models,
            pinned=message.pinned,
            prompt_cached=message.prompt_cached,
            total_tokens=message.total_tokens if isinstance(message, Assistant) else 0,
            underlying=message.underlying if isinstance(message, Assistant) else None,
            response_id=getattr(message, 'response_id', None) if isinstance(message, Assistant) else None,
            round_id=getattr(message, 'round_id', None) if isinstance(message, Assistant) else None,
            inference_number=getattr(message, 'inference_number', None) if isinstance(message, Assistant) else None,
        )


class SessionThreadModel(BaseModel):
    id: int = -1
    title: str = ''
    executor: str = ''
    api_endpoint: str = ''
    api_key: str = ''
    model: str = ''
    compression: str = ''
    temperature: float = 0.0
    stop_tokens: list[str] = Field(default_factory=list)
    output_token_len: int = 0
    current_mode: str = 'tools'
    thinking: int = 0
    compile_prompt: str = ''
    cookies: list[dict[str, Any]] = Field(default_factory=list)
    messages: list[MessageModel] = Field(default_factory=list)
    locals_dict: dict[str, Any] = Field(default_factory=dict, exclude=True, repr=False)

    # Approval flow fields
    execution_id: str = ''
    approval_response: dict[str, Any] = Field(default_factory=dict)

    # Response API conversation state (client-managed)
    previous_response_id: str = ''