"""autofe_bench: harness para comparar metodos de AutoFE com LLM sob o mesmo protocolo."""
__version__ = "0.1.0"

from .contract import AutoFEMethod, Budget, Context, FittedFE, IdentityFitted, Task  # noqa: F401
from .protocol import PROTOCOLS, Protocol, get_protocol  # noqa: F401
