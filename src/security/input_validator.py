import re

def is_input_strong(text: str):
    text = str(text).strip()

    # Remove extra spaces
    words = re.findall(r"\b\w+\b", text)

    word_count = len(words)
    char_count = len(text)

    # RULES (you can adjust later)
    if word_count < 8:
        return False, "Input is too weak: too few words"

    if char_count < 30:
        return False, "Input is too weak: too short"

    # If mostly numbers or junk
    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    if alpha_ratio < 0.6:
        return False, "Input is too weak: not enough meaningful text"

    return True, "Input is valid"