"""Helper utilities for workflow evaluation moved out of workflow_shuffling.py
包含混淆（ID/文本）和 LLM 输出解析等工具函数，供 `tasks.workflow_shuffling` 使用。
"""
import json
import re
import uuid
from typing import Any, Dict


ORDINAL_WORDS = [
    "first", "second", "third", "fourth", "fifth",
    "then", "next", "finally", "last", "before", "after",
    "step"
]


def _scrub_text(text: str) -> str:
    """Remove obvious order hints from name/description."""
    t = re.sub(r"\bstep\s*\d+\b", "step", text, flags=re.I)
    t = re.sub(r"\b(" + "|".join(ORDINAL_WORDS) + r")\b", "", t, flags=re.I)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t


def _deep_replace_step_refs(obj: Any, id_mapping: Dict[str, str]) -> Any:
    """
    Recursively replace {{step_<old>...}} to {{step_<new>...}} in strings.
    处理多种格式：
    - {{step_<id>.field}} (无空格)
    - {{ step_<id>.field }} (有空格)
    """
    if isinstance(obj, dict):
        return {k: _deep_replace_step_refs(v, id_mapping) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_replace_step_refs(x, id_mapping) for x in obj]
    if isinstance(obj, str):
        def repl(m):
            prefix = m.group(1)  # 开头的空格
            old = m.group(2)      # step_id
            tail = m.group(3)     # 后续内容
            suffix = m.group(4)   # 结尾的空格
            new = id_mapping.get(old, old)
            return "{{" + prefix + "step_" + new + tail + suffix + "}}"
        
        # 匹配两种格式：{{\s*step_(\w+)([^}]*)\s*}}
        return re.sub(r"\{\{(\s*)step_(\w+)([^}]*)(\s*)\}\}", repl, obj)
    return obj


def obfuscate_steps(steps: list) -> Dict[str, str]:
    """
    Assign non-sequential IDs (short UUID), update dependencies and template refs,
    and scrub order-hint words in names/descriptions.
    Returns: id_mapping {old_id -> new_id}
    """
    new_ids = [uuid.uuid4().hex[:8] for _ in steps]
    id_mapping = {step['step_id']: new_ids[i] for i, step in enumerate(steps)}

    for step in steps:
        old = step['step_id']
        step['step_id'] = id_mapping[old]

    for step in steps:
        # Update steps
        if 'next_steps' in step and isinstance(step['next_steps'], list):
            step['next_steps'] = [id_mapping.get(x, x) for x in step['next_steps']]
        if 'else_steps' in step and isinstance(step.get('else_steps'), list):
            step['else_steps'] = [id_mapping.get(x, x) for x in step['else_steps']]
        if 'error_handler' in step and step.get('error_handler'):
            step['error_handler'] = id_mapping.get(step['error_handler'], step['error_handler'])

        # Replace template refs in parameters or any fields
        for key in list(step.keys()):
            step[key] = _deep_replace_step_refs(step[key], id_mapping)

        # Scrub names/descriptions
        if isinstance(step.get('name'), str):
            step['name'] = _scrub_text(step['name'])
        if isinstance(step.get('description'), str):
            step['description'] = _scrub_text(step['description'])

        # Replace step IDs in the 'condition' field if present
        if 'condition' in step and isinstance(step['condition'], dict):
            step['condition'] = _deep_replace_step_refs(step['condition'], id_mapping)
        
        # Replace step IDs in the 'parameters' field if present
        if 'parameters' in step and isinstance(step['parameters'], dict):
            step['parameters'] = _deep_replace_step_refs(step['parameters'], id_mapping)
            
    return id_mapping


def parse_llm_output(text: str) -> Dict:
    """从LLM输出中提取JSON"""
    text = text.strip()
    
    # 尝试直接解析
    try:
        data = json.loads(text)
        if isinstance(data, dict) and 'workflow_steps' in data:
            return data
    except json.JSONDecodeError:
        pass
    
    # 提取JSON
    start = None
    for i, ch in enumerate(text):
        if ch in '{[':
            start = i
            break
    
    if start is None:
        raise ValueError("No JSON found in LLM output")
    
    # 匹配括号
    stack, in_string, esc = [], False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == '{':
                stack.append('{')
            elif ch == '[':
                stack.append('[')
            elif ch == '}':
                if stack and stack[-1] == '{':
                    stack.pop()
            elif ch == ']':
                if stack and stack[-1] == '[':
                    stack.pop()
            
            if not stack and i > start:
                payload = text[start:i+1]
                data = json.loads(payload)
                if isinstance(data, dict) and 'workflow_steps' in data:
                    return data
                break
    
    raise ValueError("Failed to extract valid workflow JSON")
