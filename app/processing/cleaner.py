import re


class TextCleaner:
    def clean(self, text: str) -> str:
        if not text:
            return ""

        text = self._normalize_line_endings(text)
        text = self._remove_excessive_whitespace(text)
        text = self._normalize_spaces(text)

        return text.strip()

    def _normalize_line_endings(self, text: str) -> str:
        return text.replace("\r\n", "\n").replace("\r", "\n")

    def _remove_excessive_whitespace(self, text: str) -> str:
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text

    def _normalize_spaces(self, text: str) -> str:
        return text.replace("\xa0", " ")