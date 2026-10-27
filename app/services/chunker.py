import re
from typing import List, Dict, Any, Generator

# Sentence boundary patterns (handles English . ? !, Hindi ।, newlines)
SENTENCE_SPLIT_REGEX = re.compile(r'(?<=[.?!।\n])\s+')


def split_into_sentence_chunks(text: str) -> List[str]:
    """
    Split a long user transcript into individual sentence/clause chunks.
    Works for Hindi (।), English (.?!), Hinglish, and newlines.
    """
    cleaned = text.strip()
    if not cleaned:
        return []

    # Split using sentence boundary regex
    raw_chunks = SENTENCE_SPLIT_REGEX.split(cleaned)
    chunks = [c.strip() for c in raw_chunks if c.strip()]
    return chunks if chunks else [cleaned]


def optimize_long_user_query(text: str) -> Dict[str, Any]:
    """
    Preprocess user input:
    If the user speaks a long query (5-7 lines or multiple sentences):
    1. Breaks into semantic chunks.
    2. Identifies the primary question/request chunk.
    3. Formats an optimized instruction for the LLM to deliver a direct, low-latency answer.
    """
    chunks = split_into_sentence_chunks(text)
    is_multi_sentence = len(chunks) > 1 or len(text) > 120

    if not is_multi_sentence:
        return {
            "is_chunked": False,
            "chunks_count": 1,
            "chunks": [text],
            "optimized_prompt": text,
        }

    # Identify primary inquiry chunk (prioritize chunks with question words or questions marks)
    question_keywords = ["kya", "kaise", "kyun", "kab", "kahan", "kaun", "batao", "what", "how", "why", "when", "where", "who", "tell", "explain", "help"]
    primary_chunk = None

    for c in reversed(chunks):
        if any(w in c.lower() for w in question_keywords) or c.endswith("?"):
            primary_chunk = c
            break

    if not primary_chunk:
        primary_chunk = chunks[-1]  # Usually the conclusion or direct ask is at the end

    context_summary = " ".join(chunks)[:250]
    optimized_prompt = context_summary

    return {
        "is_chunked": True,
        "chunks_count": len(chunks),
        "chunks": chunks,
        "primary_chunk": primary_chunk,
        "optimized_prompt": optimized_prompt,
    }


class SentenceStreamBuffer:
    """
    Buffers incoming LLM streaming tokens and yields complete sentences as soon as
    sentence boundary punctuation (. ? ! । \n) is encountered.
    Enables instant sentence-by-sentence TTS synthesis!
    """

    PUNCTUATIONS = {".", "?", "!", "।", "\n"}

    def __init__(self, min_chunk_words: int = 3):
        self.buffer = ""
        self.min_chunk_words = min_chunk_words

    def process_token(self, token: str) -> List[str]:
        """
        Add a token and return any completed sentence chunks ready for TTS.
        """
        self.buffer += token
        ready_sentences = []

        # Check for punctuation markers
        for p in self.PUNCTUATIONS:
            if p in self.buffer:
                # Find punctuation position
                idx = self.buffer.find(p)
                # Ensure the chunk has sufficient words to avoid premature micro-chunks
                candidate = self.buffer[: idx + 1].strip()
                words_count = len(candidate.split())

                if words_count >= self.min_chunk_words:
                    ready_sentences.append(candidate)
                    self.buffer = self.buffer[idx + 1 :].lstrip()
                    break

        return ready_sentences

    def flush(self) -> str:
        """Return any remaining text in the buffer."""
        remaining = self.buffer.strip()
        self.buffer = ""
        return remaining
