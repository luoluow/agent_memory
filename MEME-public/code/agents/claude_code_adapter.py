"""
Claude Code → OpenAI compatibility adapter.

Routes chat.completions.create() calls through the `claude -p` CLI so that
evaluation can run against the user's Claude Pro subscription without needing
an ANTHROPIC_API_KEY.

Usage:
    from agents.claude_code_adapter import ClaudeCodeAsOpenAI
    client = ClaudeCodeAsOpenAI()
    response = client.chat.completions.create(
        model="claude-code",          # or "claude-code/sonnet", "claude-code/opus"
        messages=[...],
        temperature=0,
        max_tokens=500,
    )
    print(response.choices[0].message.content)

Model string format:
    "claude-code"            → use claude CLI default model
    "claude-code/sonnet"     → pass --model sonnet to claude CLI
    "claude-code/opus"       → pass --model opus to claude CLI
    "claude-code/<full-id>"  → pass --model <full-id> to claude CLI
"""

import subprocess


class _Message:
    def __init__(self, content: str):
        self.content = content
        self.role = "assistant"
        self.tool_calls = None


class _Choice:
    def __init__(self, message: _Message):
        self.message = message


class _Response:
    def __init__(self, content: str):
        self.choices = [_Choice(_Message(content))]


class _Completions:
    def create(self, model: str, messages, temperature=0, max_tokens=500, **kwargs) -> _Response:
        system_parts = []
        user_parts = []

        for msg in messages:
            role = msg["role"] if isinstance(msg, dict) else getattr(msg, "role", "")
            content = msg["content"] if isinstance(msg, dict) else getattr(msg, "content", "")
            if role == "system":
                system_parts.append(content)
            elif role == "user":
                user_parts.append(content)
            elif role == "assistant":
                user_parts.append(f"Assistant: {content}")

        prompt = "\n\n".join(user_parts)

        cmd = [
            "claude", "-p",
            "--output-format", "text",
            "--tools", "",
            "--no-session-persistence",
        ]

        # Extract optional sub-model from "claude-code/sonnet" style strings
        if "/" in model:
            sub_model = model.split("/", 1)[1]
            cmd.extend(["--model", sub_model])

        if system_parts:
            cmd.extend(["--system-prompt", "\n".join(system_parts)])

        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=180,
        )

        if result.returncode != 0 and not result.stdout.strip():
            raise RuntimeError(
                f"claude CLI failed (exit {result.returncode}): {result.stderr.strip()}"
            )

        return _Response(result.stdout.strip())


class _Chat:
    def __init__(self):
        self.completions = _Completions()


class ClaudeCodeAsOpenAI:
    """Drop-in replacement for OpenAI() that routes through the claude CLI."""

    def __init__(self):
        self.chat = _Chat()
