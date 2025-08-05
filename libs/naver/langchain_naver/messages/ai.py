import logging
from typing import Union, Optional

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
)

logger = logging.getLogger(__name__)


class ClovaXAIMessage(AIMessage):
    """Message from an AI."""
    reasoning_content: Optional[Union[str, list[Union[str, dict]]]] = None


ClovaXAIMessage.model_rebuild()


class ClovaXAIMessageChunk(ClovaXAIMessage, AIMessageChunk):
    """Message chunk from an AI."""
