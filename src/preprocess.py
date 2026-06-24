import re

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were",
    "to", "of", "and", "in", "on", "for", "with",
    "at", "by", "from", "that", "this", "it"
}

def clean_text(text):
    text = str(text).lower()

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    words = text.split()

    words = [word for word in words if word not in STOP_WORDS]

    return " ".join(words)