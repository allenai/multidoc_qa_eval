import json
import logging
from typing import Any, Dict, Optional

from langchain_openai import ChatOpenAI

LOGGER = logging.getLogger(__name__)


def extract_json_from_response(response: str) -> Optional[Dict[str, Any]]:
    json_start = response.find("{")
    json_end = response.rfind("}") + 1
    if json_start == -1 or json_end == -1:
        return None

    try:
        return json.loads(response[json_start:json_end])
    except json.JSONDecodeError:
        LOGGER.warning(
            f"Could not decode JSON from response: {response[json_start:json_end]}"
        )
        return None


def run_chatopenai(
    model_name: str,
    system_prompt: Optional[str],
    user_prompt: str,
    json_mode: bool = False,
    **chat_kwargs,
) -> str:
    chat_kwargs["temperature"] = chat_kwargs.get("temperature", 0)
    llm = ChatOpenAI(
        model=model_name,
        model_kwargs=(
            ({"response_format": {"type": "json_object"}}) if json_mode else dict()
        ),
        **chat_kwargs,
    )

    msgs = (
        [("system", system_prompt), ("human", user_prompt)]
        if system_prompt is not None
        else [("human", user_prompt)]
    )

    resp = llm.invoke(msgs).content

    # Langchain APIs are weird and technically don't promise to return a string
    assert isinstance(resp, str)

    return resp
