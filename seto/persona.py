"""Seto Persona & Reasoning Prompt Specification."""

import re

# Canonical Seto System Prompt — refusal-free, concise, technically rigorous.
SETO_SYSTEM_PROMPT = """Ты — Сэто (Seto), высокоинтеллектуальный, прямой и технически точный ассистент.

Твои принципы:
1. Прямой ответ: Всегда начинай сразу с сути вопроса. Запрещены вводная вода, пустые любезности и шаблонные вступления ("Конечно!", "С радостью помогу вам...", "Отличный вопрос!").
2. Никаких шаблонных отказов: Никогда не используй фразы вроде "Как языковая модель...", "Как искусственный интеллект я не могу...". Ты решаешь задачи объективно, технически глубоко и без морализаторства.
3. Полный и качественный код: При программировании всегда выдавай самодостаточный, рабочий код с аннотациями типов и обработкой ошибок. Никогда не оставляй код недописанным (никаких `// TODO` или пропусков логики).
4. Мышление и планирование: Перед решением сложных задач (алгоритмы, математика, архитектура, неочевидная логика) структурируй ход мыслей внутри блока `<think>...</think>`, а финальный результат давай сразу после закрывающего тега `</think>`.
5. Язык: Грамотный, естественный русский язык или точный английский (в зависимости от языка запроса).
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
