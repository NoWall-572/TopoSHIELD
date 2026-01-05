# --- START OF FILE sandbox/generate.py ---

from typing import Dict, Optional, Any
from openai import OpenAI, AsyncOpenAI

def _normalize_base_url(u: Optional[str]) -> Optional[str]:
    """
    Normalizes the OpenAI-compatible base URL to ensure proper formatting.
    Ensures the URL ends with '/v1' and strips specific endpoint paths.
    """
    u = (u or "").strip()
    if not u:
        return u

    u = u.rstrip("/")
    for suffix in ["/chat/completions", "/v1/chat/completions", "/responses"]:
        if u.endswith(suffix):
            u = u[: -len(suffix)].rstrip("/")
            break

    if u.endswith("/v1"):
        return u
    return u + "/v1"


def generate_with_gpt(
    prompt: str,
    model: Dict[str, Any],
    print_prompt: bool = False,
) -> str:
    """
    Synchronous generation using the OpenAI client.
    
    Args:
        prompt (str): The input prompt for the model.
        model (Dict): Configuration dictionary containing 'api_key', 'base_url', and 'name'.
        print_prompt (bool): If True, prints the prompt to stdout for debugging.

    Returns:
        str: The generated text content. Returns an empty string on failure.
    """
    if print_prompt:
        print(prompt)

    base_url = _normalize_base_url(model.get("base_url"))

    try:
        client = OpenAI(api_key=model["api_key"], base_url=base_url)
        
        generated_text = client.chat.completions.create(
            model=model["name"],
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        return str(generated_text.choices[0].message.content)
    except Exception as e:
        print(f"[Generate Error] {e}")
        return ""


async def agenerate(
    prompt: str,
    model: str = "gpt-4o",
    aclient: Optional[AsyncOpenAI] = None,
) -> str:
    """
    Asynchronous generation using a pre-initialized AsyncOpenAI client.

    Args:
        prompt (str): The input prompt.
        model (str): The model name (e.g., 'gpt-4o-mini').
        aclient (AsyncOpenAI): An active AsyncOpenAI client instance.

    Returns:
        str: The generated text content or an error identifier string.
    """
    if aclient is None:
        return "[[Error: No Async Client Provided]]"

    try:
        completion = await aclient.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        generated_text = str(completion.choices[0].message.content)
        return generated_text
    except Exception as e:
        return f"[[API_ERROR: {e}]]"

# --- END OF FILE sandbox/generate.py ---