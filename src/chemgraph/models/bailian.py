"""Load Alibaba Bailian (百炼) models using LangChain."""

import os
from getpass import getpass
from langchain_openai import ChatOpenAI
from chemgraph.utils.logging_config import setup_logger

logger = setup_logger(__name__)

BAILIAN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def load_bailian_model(
    model_name: str,
    temperature: float,
    api_key: str = None,
    base_url: str = None,
) -> ChatOpenAI:
    """Load an Alibaba Bailian (百炼) chat model via its OpenAI-compatible API.

    Parameters
    ----------
    model_name : str
        The name of the Bailian model (e.g. "qwen-max", "qwen-plus").
    temperature : float
        Sampling temperature. Use 0.0 for deterministic tool-calling.
    api_key : str, optional
        DashScope API key. Falls back to the ``DASHSCOPE_API_KEY`` environment
        variable and prompts interactively if neither is available.
    base_url : str, optional
        Override the default Bailian base URL. Defaults to
        ``https://dashscope.aliyuncs.com/compatible-mode/v1``.

    Returns
    -------
    ChatOpenAI
        A LangChain ``ChatOpenAI`` instance pointed at the Bailian endpoint.
    """
    if api_key is None:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            logger.info("DashScope API key not found in environment variables.")
            api_key = getpass("Please enter your DashScope (百炼) API key: ")
            os.environ["DASHSCOPE_API_KEY"] = api_key

    resolved_base_url = base_url or BAILIAN_BASE_URL

    try:
        logger.info(f"Loading Bailian model '{model_name}' from {resolved_base_url}")
        llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
            base_url=resolved_base_url,
            max_tokens=4000,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        logger.info(f"Bailian model '{model_name}' loaded successfully")
        return llm
    except Exception as e:
        if "AuthenticationError" in str(e) or "invalid_api_key" in str(e):
            logger.warning("Invalid DashScope API key.")
            api_key = getpass("Please enter a valid DashScope (百炼) API key: ")
            os.environ["DASHSCOPE_API_KEY"] = api_key
            return load_bailian_model(model_name, temperature, api_key, base_url)
        logger.error(f"Error loading Bailian model: {str(e)}")
        raise
