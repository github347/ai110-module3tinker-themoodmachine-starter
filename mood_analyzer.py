# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

import re

from typing import List, Dict, Tuple, Optional

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens the model can work with.

        TODO: Improve this method.

        Right now, it does the minimum:
          - Strips leading and trailing whitespace
          - Converts everything to lowercase
          - Splits on spaces

        Ideas to improve:
          - Remove punctuation
          - Handle simple emojis separately (":)", ":-(", "🥲", "😂")
          - Normalize repeated characters ("soooo" -> "soo")
        """
        if not text:
          return []

        cleaned = text.strip().lower()

        emoticon_map = {
          ":)": "smile_emoji",
          ":-)": "smile_emoji",
          ":(": "sad_emoji",
          ":-(": "sad_emoji",
          ":d": "laugh_emoji",
          ":-d": "laugh_emoji",
        }

        for emoticon, token in emoticon_map.items():
          cleaned = cleaned.replace(emoticon, f" {token} ")

        emoji_map = {
          "😂": "laugh_emoji",
          "🥲": "sad_emoji",
          "💀": "dead_emoji",
          "🙃": "sarcasm_emoji",
          "🌎": "world_emoji",
        }

        for emoji, token in emoji_map.items():
          cleaned = cleaned.replace(emoji, f" {token} ")

        cleaned = cleaned.replace("-", " ")
        cleaned = re.sub(r"[^a-z0-9_\s']", " ", cleaned)

        raw_tokens = cleaned.split()

        def normalize_repeated_characters(token: str) -> str:
          return re.sub(r"(.)\1{2,}", r"\1\1", token)

        tokens = [normalize_repeated_characters(token) for token in raw_tokens if token]
        return tokens

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def score_text(self, text: str) -> int:
        """
        Compute a numeric "mood score" for the given text.

        Positive words increase the score.
        Negative words decrease the score.

        TODO: You must choose AT LEAST ONE modeling improvement to implement.
        For example:
          - Handle simple negation such as "not happy" or "not bad"
          - Count how many times each word appears instead of just presence
          - Give some words higher weights than others (for example "hate" < "annoyed")
          - Treat emojis or slang (":)", "lol", "💀") as strong signals
        """
        tokens = self.preprocess(text)

        negation_tokens = {
          "not",
          "no",
          "never",
          "don't",
          "dont",
          "can't",
          "cant",
          "won't",
          "wont",
          "isn't",
          "isnt",
          "wasn't",
          "wasnt",
        }

        emoji_weights = {
          "smile_emoji": 1,
          "laugh_emoji": 1,
          "sad_emoji": -1,
          "dead_emoji": -1,
          "sarcasm_emoji": -1,
        }

        score = 0

        for index, token in enumerate(tokens):
          if token in emoji_weights:
            score += emoji_weights[token]
            continue

          prev_token = tokens[index - 1] if index > 0 else ""
          is_negated = prev_token in negation_tokens

          if token in self.positive_words:
            score += -1 if is_negated else 1
          elif token in self.negative_words:
            score += 1 if is_negated else -1

        return score

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the numeric score for a piece of text into a mood label.

        The default mapping is:
          - score > 0  -> "positive"
          - score < 0  -> "negative"
          - score == 0 -> "neutral"

        TODO: You can adjust this mapping if it makes sense for your model.
        For example:
          - Use different thresholds (for example score >= 2 to be "positive")
          - Add a "mixed" label for scores close to zero
        Just remember that whatever labels you return should match the labels
        you use in TRUE_LABELS in dataset.py if you care about accuracy.
        """
        score = self.score_text(text)

        if score > 0:
            return "positive"
        elif score < 0:
            return "negative"
        else:
            tokens = self.preprocess(text)
            has_positive = any(t in self.positive_words for t in tokens)
            has_negative = any(t in self.negative_words for t in tokens)
            if has_positive and has_negative:
                return "mixed"
            else:
                return "neutral"

    # ---------------------------------------------------------------------
    # Explanations (optional but recommended)
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label.

        TODO:
          - Look at the tokens and identify which ones counted as positive
            and which ones counted as negative.
          - Show the final score.
          - Return a short human readable explanation.

        Example explanation (your exact wording can be different):
          'Score = 2 (positive words: ["love", "great"]; negative words: [])'

        The current implementation is a placeholder so the code runs even
        before you implement it.
        """
        tokens = self.preprocess(text)

        negation_tokens = {
            "not",
            "no",
            "never",
            "don't",
            "dont",
            "can't",
            "cant",
            "won't",
            "wont",
            "isn't",
            "isnt",
            "wasn't",
            "wasnt",
        }

        emoji_weights = {
            "smile_emoji": 1,
            "laugh_emoji": 1,
            "sad_emoji": -1,
            "dead_emoji": -1,
            "sarcasm_emoji": -1,
        }

        positive_hits: List[str] = []
        negative_hits: List[str] = []
        emoji_hits: List[str] = []

        for index, token in enumerate(tokens):
            if token in emoji_weights:
                emoji_hits.append(token)
                continue

            prev_token = tokens[index - 1] if index > 0 else ""
            is_negated = prev_token in negation_tokens

            if token in self.positive_words:
                if is_negated:
                    negative_hits.append(f"not {token}")
                else:
                    positive_hits.append(token)
            elif token in self.negative_words:
                if is_negated:
                    positive_hits.append(f"not {token}")
                else:
                    negative_hits.append(token)

        score = self.score_text(text)

        return (
            f"Score = {score} "
            f"(positive: {positive_hits or '[]'}, "
            f"negative: {negative_hits or '[]'}, "
            f"emojis: {emoji_hits or '[]'})"
        )
