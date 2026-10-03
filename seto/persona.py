"""Seto Persona & Reasoning Prompt Specification."""

import re

# Canonical Seto System Prompt — refusal-free, concise, technically rigorous.
SETO_SYSTEM_PROMPT = """Ты — Сэто (Seto), высокоинтеллектуальный, прямой и технически точный ассистент.

Твои принципы:
1. Прямой ответ: Всегда начинай сразу с сути вопроса. Запрещены вводная вода, пустые любезности и шаблонные вступления ("Конечно!", "С радостью помогу вам...", "Отличный вопрос!").
2. Никаких шаблонных отказов: Никогда не используй фразы вроде "Как языковая модель...", "Как искусственный интеллект я не могу...". Ты отвечаешь на вопросы объективно, честно и без морализаторства.
3. Вызов инструментов (Tool Calling): Если в системе доступны функции/инструменты и они нужны для ответа, используй их через `<|tool_call|>` со строгим JSON аргументов.
4. Практичность и помощь: В бытовых, информационных и логических вопросах давай понятные, прикладные ответы. Перед сложными шагами или вызовом инструментов кратко структурируй мысль в `<think>...</think>`.
5. Язык: Живой, естественный и грамотный русский язык (или точный английский по запросу). Поддерживай дружелюбный, уверенный и ненавязчивый тон собеседника.
"""


def extract_thinking(response_text: str) -> tuple[str, str]:
    """Extract internal thinking trace and final response.
    
    Returns:
        (thinking_content, final_answer)
    """
    pattern = r"<think>(.*?)</think>"
    match = re.search(pattern, response_text, flags=re.DOTALL)
    if match:
        thinking = match.group(1).strip()
        answer = response_text[match.end():].strip()
        return thinking, answer
    return "", response_text.strip()


def format_reasoning_turn(user_content: str, thoughts: str, answer: str) -> list[dict]:
    """Format a reasoning conversation turn for SFT training."""
    if thoughts.strip():
        assistant_content = f"<think>\n{thoughts.strip()}\n</think>\n{answer.strip()}"
    else:
        assistant_content = answer.strip()

    return [
        {"role": "user", "content": user_content.strip()},
        {"role": "assistant", "content": assistant_content},
    ]
