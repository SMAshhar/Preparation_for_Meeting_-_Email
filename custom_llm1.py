from crewai import BaseLLM
from typing import Any, Dict, List, Optional, Union
import json
import re
import requests
import os

from json_output import extract_json_from_text

# TypeScript-style placeholders are not valid JSON (models sometimes echo the rubric).
_SCHEMA_VALUE_PLACEHOLDER = re.compile(
    r":\s*\b(float|string|boolean|int)\b\s*([,}\]]|$)",
    re.IGNORECASE,
)


def _raise_if_schema_placeholder_echo(text: str) -> None:
    if text and _SCHEMA_VALUE_PLACEHOLDER.search(text):
        raise ValueError(
            "LLM echoed schema placeholders (float/string/boolean/int) instead of real JSON values; retrying."
        )


def _truncate_chat_template_junk(text: str) -> str:
    """Remove continuation after common chat-model markers (prevents trailing JSON parse errors)."""
    if not text:
        return text
    for marker in ("<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|assistant|>", "<|user|>"):
        if marker in text:
            text = text.split(marker, 1)[0]
    return text


def _ollama_assistant_text(message: dict) -> str:
    """
    Qwen 3.x / some Ollama models put chain-of-thought in `thinking` and leave `content` empty.
    CrewAI treats empty assistant text as failure ('None or empty response').
    Prefer non-empty `content` (final JSON); only fall back to thinking/reasoning when content is blank.
    """
    if not isinstance(message, dict):
        return ""
    c = message.get("content")
    if isinstance(c, str) and c.strip():
        return c.strip()
    parts: List[str] = []
    for key in ("thinking", "reasoning", "thought"):
        val = message.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
    return "\n\n".join(parts)


def _sanitize_structured_llm_text(text: Optional[str]) -> str:
    """
    CrewAI validates task outputs with Pydantic model_validate_json on the full string.
    Local models often append <|endoftext|> and extra tokens after valid JSON — strip to one JSON value.
    """
    if text is None or not isinstance(text, str) or not text.strip():
        return ""
    text = _truncate_chat_template_junk(text)
    if "{" not in text:
        return text
    extracted = extract_json_from_text(text)
    if extracted:
        _raise_if_schema_placeholder_echo(extracted)
        return extracted
    start = text.find("{")
    if start == -1:
        return text
    try:
        decoder = json.JSONDecoder()
        _, end = decoder.raw_decode(text[start:])
        out = text[start : start + end]
        _raise_if_schema_placeholder_echo(out)
        return out
    except json.JSONDecodeError:
        _raise_if_schema_placeholder_echo(text)
        return text


class CustomLLM1(BaseLLM):
    def __init__(self, model: str, api_key: str, endpoint: str, temperature: Optional[float] = None, timeout: Optional[int] = None):
        super().__init__(model=model, temperature=temperature)
        self.api_key = api_key or ""
        self.endpoint = endpoint
        # Allow timeout to be set per instance, or use environment variable, or default to 120
        self.timeout = timeout or int(os.getenv("CUSTOM_LLM_TIMEOUT", "120"))

    def call(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: Optional[List[dict]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
        **kwargs  # Accept any additional args from the caller (e.g. from_task)
    ) -> Union[str, Any]:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        # Respect explicit 'stream' kwarg, default to False to avoid streaming
        stream_value = kwargs.pop("stream", False)

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": stream_value,
        }

        # include tools if function calling is supported
        if tools and self.supports_function_calling():
            payload["tools"] = tools

        # If you want to allow a few other top-level options, whitelist them here:
        for k in ("max_tokens", "top_p", "n"):
            if k in kwargs:
                payload[k] = kwargs[k]

        # Cap completion tokens (Ollama num_predict). Reduces runaway generation; raise if aggregation JSON is truncated.
        num_predict_raw = os.getenv("CUSTOM_LLM_NUM_PREDICT", "4096").strip()
        if num_predict_raw and num_predict_raw not in ("0", "-1"):
            try:
                np = int(num_predict_raw)
                if np > 0:
                    payload.setdefault("options", {})["num_predict"] = np
            except ValueError:
                pass
                
        # Enforce GPU usage via num_gpu option (-1 means all layers on GPU)
        num_gpu_raw = os.getenv("CUSTOM_LLM_NUM_GPU", "-1").strip()
        if num_gpu_raw:
            try:
                ng = int(num_gpu_raw)
                payload.setdefault("options", {})["num_gpu"] = ng
            except ValueError:
                pass

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Check if this is the gpt-oss model - use 600 seconds timeout
        is_gpt_oss = "gpt-oss" in self.model.lower() or "SimonPu/gpt-oss" in self.model
        
        # Check if this is an aggregation task - use longer timeout
        # CrewAI may pass task info in kwargs (e.g., from_task)
        timeout_value = self.timeout
        if is_gpt_oss:
            # Floor at 600s for very slow models, but allow CUSTOM_LLM_TIMEOUT to go higher (e.g. 1800).
            timeout_value = max(600, self.timeout)
        elif kwargs.get("from_task") and "aggregation" in str(kwargs.get("from_task", "")).lower():
            # Use longer timeout for aggregation tasks (3x default)
            timeout_value = int(os.getenv("CUSTOM_LLM_AGGREGATION_TIMEOUT", str(self.timeout * 3)))
        elif "aggregation" in str(messages).lower():
            # Fallback: check if aggregation is mentioned in messages
            timeout_value = int(os.getenv("CUSTOM_LLM_AGGREGATION_TIMEOUT", str(self.timeout * 3)))

        try:
            response = requests.post(
                self.endpoint,
                headers=headers,
                json=payload,
                timeout=timeout_value  # Configurable timeout, longer for aggregation tasks
            )
            response.raise_for_status()
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Failed to connect to LLM server at {self.endpoint}. "
                f"Make sure Ollama is running (ollama serve) or your LLM server is accessible. "
                f"Error: {str(e)}"
            )
        except requests.exceptions.Timeout as e:
            raise TimeoutError(
                f"Request to LLM server timed out after {timeout_value} seconds. "
                f"The model might be too slow or the server is overloaded. "
                f"For large models or aggregation tasks, increase CUSTOM_LLM_TIMEOUT or CUSTOM_LLM_AGGREGATION_TIMEOUT in your .env file. "
                f"If the model runs very long on each token, set CUSTOM_LLM_NUM_PREDICT (e.g. 4096) to cap output length. "
                f"Error: {str(e)}"
            )
        except requests.exceptions.HTTPError as e:
            raise ValueError(
                f"HTTP error from LLM server: {e.response.status_code if hasattr(e, 'response') else 'unknown'}. "
                f"Check your model name and endpoint configuration. "
                f"Error: {str(e)}"
            )

        # Try to parse a single JSON document first; fall back to streaming/newline parsing.
        try:
            result = response.json()
        except Exception:
            text = response.text or ""
            text = text.strip()
            parsed = None
            if text:
                # Try SSE / newline-delimited JSON lines
                for line in text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("data:"):
                        line = line[len("data:"):].strip()
                    try:
                        parsed = json.loads(line)
                        break
                    except Exception:
                        continue

            # If still nothing, try to decode the first JSON object from concatenated JSON text
            if parsed is None and text:
                try:
                    decoder = json.JSONDecoder()
                    parsed_obj, _ = decoder.raw_decode(text)
                    parsed = parsed_obj
                except Exception:
                    parsed = None

            if parsed is None:
                # final fallback: return raw text so caller can inspect it
                return _sanitize_structured_llm_text(text)

            result = parsed

        # 1) OpenAI-like response (also used by some Ollama-compatible proxies)
        try:
            if isinstance(result, dict) and "choices" in result and len(result["choices"]) > 0:
                first = result["choices"][0]
                if isinstance(first, dict):
                    if "message" in first and isinstance(first["message"], dict):
                        combined = _ollama_assistant_text(first["message"])
                        if combined:
                            return _sanitize_structured_llm_text(combined)
                    if "text" in first and isinstance(first["text"], str):
                        return _sanitize_structured_llm_text(first["text"])
        except Exception:
            pass

        # 2) Ollama-like response shapes (native /api/chat uses top-level "message")
        if isinstance(result, dict):
            # {"message": {"content"?, "thinking"?, "role": "assistant"}} — content may be empty for Qwen 3.x
            if "message" in result and isinstance(result["message"], dict):
                combined = _ollama_assistant_text(result["message"])
                if combined:
                    return _sanitize_structured_llm_text(combined)
            # Alternative formats
            if "result" in result and isinstance(result["result"], str):
                return _sanitize_structured_llm_text(result["result"])
            if "output" in result and isinstance(result["output"], str):
                return _sanitize_structured_llm_text(result["output"])
            # Ollama sometimes returns {"choices":[{"text":"..."}]}
            if "choices" in result and isinstance(result["choices"], list) and len(result["choices"]) > 0:
                ch = result["choices"][0]
                if isinstance(ch, dict) and "text" in ch:
                    return _sanitize_structured_llm_text(ch["text"])

        # 3) Fallback: return a stringified JSON
        return str(result)

    def supports_function_calling(self) -> bool:
        # CrewAI passes `tools` when True; Ollama + local models often return empty `content` in that mode.
        # Enable only if you use tool-calling: CUSTOM_LLM_ENABLE_TOOLS=1
        return os.getenv("CUSTOM_LLM_ENABLE_TOOLS", "").strip().lower() in ("1", "true", "yes")

    def get_context_window_size(self) -> int:
        return 8192
    


