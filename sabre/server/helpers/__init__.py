"""
LLM-callable helper functions.

These classes and methods are exposed to the LLM in the Python runtime.
"""
from sabre.server.helpers.bash import Bash, BashResult
from sabre.server.helpers.search import Search
from sabre.server.helpers.web import Web, download
from sabre.server.helpers.llm_call import LLMCall, run_async_from_sync
from sabre.server.helpers.llm_bind import LLMBind
from sabre.server.helpers.coerce import Coerce
from sabre.server.helpers.llm_list_bind import LLMListBind
from sabre.server.helpers.pandas_bind import PandasBind

__all__ = [
    # Stateless helpers
    'Bash',
    'BashResult',
    'Search',
    'Web',
    'download',
    # Stateful helpers (need runtime context)
    'LLMCall',
    'LLMBind',
    'Coerce',
    'LLMListBind',
    'PandasBind',
    # Utilities
    'run_async_from_sync',
]
