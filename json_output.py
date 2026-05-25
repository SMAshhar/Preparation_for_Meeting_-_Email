import re
import json


def extract_json_from_text(s: str) -> str | None:
    """
    Extract JSON from text that may contain markdown code blocks or other formatting.
    Strips markdown code blocks (```json, ```, etc.) and extracts clean JSON.
    Also attempts to fix common JSON issues like truncated strings or missing closing brackets.
    """
    if not s:
        return None
    
    # Strip the string first
    s = s.strip()
    
    # Remove markdown code blocks (```json, ```, etc.)
    # Pattern matches: ```json, ```JSON, ```, etc.
    s = re.sub(r'^```(?:json|JSON)?\s*\n?', '', s, flags=re.MULTILINE)
    s = re.sub(r'\n?```\s*$', '', s, flags=re.MULTILINE)
    s = s.strip()
    
    # Find first JSON opening char
    start = None
    opening = closing = None
    
    for i, ch in enumerate(s):
        if ch in "{[":
            start = i
            opening = ch
            closing = "}" if ch == "{" else "]"
            break
    
    if start is None:
        return None

    # Track depth to find matching closing bracket
    depth = 0
    in_string = False
    escape = False
    last_valid_pos = start
    
    for i in range(start, len(s)):
        ch = s[i]
        
        if escape:
            escape = False
            continue
            
        if ch == "\\":
            escape = True
            continue
            
        if ch == '"' and not escape:
            in_string = not in_string
            continue
            
        if in_string:
            continue
            
        if ch == opening:
            depth += 1
            last_valid_pos = i
        elif ch == closing:
            depth -= 1
            if depth == 0:
                # Found complete JSON
                json_str = s[start : i + 1]
                return json_str.strip()
            last_valid_pos = i
    
    # If we didn't find a complete JSON, try to fix it
    # This handles truncated JSON responses
    if depth > 0:
        # Try to extract up to the last valid position and close brackets
        partial_json = s[start:last_valid_pos + 1]
        
        # Try to fix by closing unclosed strings and brackets
        fixed_json = _try_fix_truncated_json(partial_json, opening, closing, depth)
        if fixed_json:
            return fixed_json
    
    return None


def _try_fix_truncated_json(partial: str, opening: str, closing: str, depth: int) -> str | None:
    """
    Attempt to fix truncated JSON by closing unclosed strings and brackets.
    """
    try:
        # First, try to close any unclosed strings
        in_string = False
        escape = False
        fixed = list(partial)
        
        for i, ch in enumerate(fixed):
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"' and not escape:
                in_string = not in_string
        
        # If we're still in a string, close it
        if in_string:
            fixed.append('"')
        
        # Close any unclosed brackets
        for _ in range(depth):
            fixed.append(closing)
        
        fixed_str = ''.join(fixed)
        
        # Try to parse it
        json.loads(fixed_str)
        return fixed_str
    except:
        return None


def parse_json_with_fallback(text: str) -> dict | None:
    """
    Parse JSON with multiple fallback strategies for malformed JSON.
    Returns the parsed dict or None if all strategies fail.
    """
    if not text:
        return None
    
    strategies = [
        # Strategy 1: Direct parse
        lambda t: json.loads(t),
        # Strategy 2: Extract JSON from text
        lambda t: json.loads(extract_json_from_text(t) or t),
        # Strategy 3: Try to fix common issues
        lambda t: _parse_with_repairs(t),
    ]
    
    for strategy in strategies:
        try:
            return strategy(text)
        except (json.JSONDecodeError, TypeError, AttributeError):
            continue
    
    return None


def _parse_with_repairs(text: str) -> dict:
    """
    Attempt to repair common JSON issues before parsing.
    """
    # Remove trailing commas before closing brackets
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    
    # Try to close unclosed strings at the end
    if text.count('"') % 2 != 0:
        # Find the last quote and see if we need to close
        last_quote_pos = text.rfind('"')
        if last_quote_pos > 0 and text[last_quote_pos - 1] != '\\':
            # Check if we're in a string context
            before_last = text[:last_quote_pos]
            quote_count_before = before_last.count('"') - before_last.count('\\"')
            if quote_count_before % 2 == 1:
                # We're in a string, try to close it
                text = text[:last_quote_pos + 1] + '"' + text[last_quote_pos + 1:]
    
    return json.loads(text)