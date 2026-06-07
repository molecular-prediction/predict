import os
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - handled gracefully at runtime
    OpenAI = None


logger = logging.getLogger(__name__)


@dataclass
class SmileJudgement:
    smile: str
    judgment: str
    model: str
    status: str = "ok"
    error: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


class OpenAILLMProvider:
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        if OpenAI is None:
            raise RuntimeError("openai package is not installed")

        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)
        self.model = model or _get_env("LLM_MODEL_NAME", "OPENAI_MODEL", default="gpt-4.1-mini")
        self.judge_prompt = _get_env("LLM_JUDGE_PROMPT", "OPENAI_JUDGE_PROMPT", default="")
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self._token_lock = threading.Lock()
        logger.info("LLM provider initialized: model=%s base_url=%s", self.model, base_url or "default")

    @classmethod
    def from_env(cls) -> Optional["OpenAILLMProvider"]:
        _load_dotenv_file()
        api_key = _get_env("LLM_API_KEY", "OPENAI_API_KEY")
        if not api_key:
            return None

        base_url = _get_env("LLN_BASE_URL", "LLM_BASE_URL", "OPENAI_BASE_URL")
        model = _get_env("LLM_MODEL_NAME", "OPENAI_MODEL")
        try:
            return cls(model=model, api_key=api_key, base_url=base_url)
        except RuntimeError:
            return None

    def judge_smiles(self, smiles: str, prompt: Optional[str] = None) -> SmileJudgement:
        active_prompt = self.judge_prompt if prompt is None else prompt
        logger.info("LLM judging started: model=%s smiles=%s", self.model, smiles)
        messages = []
        if active_prompt:
            messages.append({"role": "system", "content": active_prompt})
        messages.append({"role": "user", "content": smiles})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        judgment = ""
        if getattr(response, "choices", None):
            choice = response.choices[0]
            message = getattr(choice, "message", None)
            judgment = getattr(message, "content", "") or ""

        # 记录 token 消耗（并发调用下用锁保护累计计数器）
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        with self._token_lock:
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            cumulative = self.total_prompt_tokens + self.total_completion_tokens

        logger.info(
            "LLM judging finished: model=%s tokens(in=%d out=%d) cum=%d smiles=%s",
            self.model, prompt_tokens, completion_tokens, cumulative, smiles,
        )
        logger.debug(
            "[LLM] 本次 tokens: %d入 %d出 | 累计消耗: %d tokens | 模型: %s",
            prompt_tokens,
            completion_tokens,
            cumulative,
            self.model,
        )

        return SmileJudgement(
            smile=smiles,
            judgment=judgment,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    @property
    def total_tokens(self) -> int:
        """累计 token 消耗总量。"""
        return self.total_prompt_tokens + self.total_completion_tokens


def _get_env(*names: str, default: Optional[str] = None) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def _load_dotenv_file() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return

    try:
        raw_lines = env_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        i = 0
        while i < len(raw_lines):
            line = raw_lines[i].strip()
            i += 1
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            key, value = line.split("=", 1)
            key = key.strip()
            if not key:
                continue

            # 处理续行：行尾的 \ 表示下一行是当前值的延续
            while value.endswith("\\") and i < len(raw_lines):
                value = value[:-1]  # 去掉行尾的续行符
                continuation = raw_lines[i].strip()
                i += 1
                value += "\n" + continuation

            value = value.strip().strip("'").strip('"')
            current = os.environ.get(key)
            if current is not None and current.strip():
                continue
            os.environ[key] = value
    except OSError:
        return
