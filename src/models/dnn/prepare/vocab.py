
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable


class Vocab:
    def __init__(
        self,
        tokens: Iterable[str] | None = None,
        *,
        specials: Iterable[str] | None = None,
        unk_token: str | None = "<unk>",
    ):
        self._token_to_index: dict[str, int] = {}
        self._index_to_token: list[str] = []
        self.frequencies: Counter[str] = Counter()
        self.specials: list[str] = []
        self.unk_token = unk_token

        ordered_specials: list[str] = []
        if unk_token is not None:
            ordered_specials.append(unk_token)
        if specials is not None:
            for token in specials:
                if token not in ordered_specials:
                    ordered_specials.append(token)

        for token in ordered_specials:
            self._add_token(token, count_frequency=False)
            self.specials.append(token)

        if tokens is not None:
            self.add_many(tokens)

    def __len__(self) -> int:
        return len(self._index_to_token)

    def __contains__(self, token: str) -> bool:
        return token in self._token_to_index

    def __iter__(self):
        return iter(self._index_to_token)

    def __getitem__(self, token: str) -> int:
        return self.lookup_index(token)

    def _add_token(self, token: str, *, count_frequency: bool = True) -> int:
        if token not in self._token_to_index:
            index = len(self._index_to_token)
            self._token_to_index[token] = index
            self._index_to_token.append(token)
        if count_frequency:
            self.frequencies[token] += 1
        return self._token_to_index[token]

    def add(self, token: str) -> int:
        return self._add_token(token)

    def add_many(self, tokens: Iterable[str]) -> list[int]:
        return [self._add_token(token) for token in tokens]

    def lookup_index(self, token: str) -> int:
        if token in self._token_to_index:
            return self._token_to_index[token]
        if self.unk_token is not None:
            return self._token_to_index[self.unk_token]
        raise KeyError(f"Unknown token: {token}")

    def lookup_token(self, index: int) -> str:
        try:
            return self._index_to_token[index]
        except IndexError as error:
            raise KeyError(f"Unknown index: {index}") from error

    def encode(self, tokens: Iterable[str]) -> list[int]:
        return [self.lookup_index(token) for token in tokens]

    def decode(self, indices: Iterable[int], *, skip_specials: bool = False) -> list[str]:
        decoded = [self.lookup_token(index) for index in indices]
        if skip_specials:
            special_tokens = set(self.specials)
            decoded = [token for token in decoded if token not in special_tokens]
        return decoded

    def to_dict(self) -> dict[str, object]:
        return {
            "tokens": list(self._index_to_token),
            "frequencies": dict(self.frequencies),
            "specials": list(self.specials),
            "unk_token": self.unk_token,
        }

    @classmethod
    def from_dict(cls, state: dict[str, object]) -> "Vocab":
        vocab = cls(specials=state.get("specials", []), unk_token=state.get("unk_token", "<unk>"))

        stored_tokens = state.get("tokens", [])
        for token in stored_tokens:
            if token not in vocab:
                vocab._add_token(token, count_frequency=False)

        frequencies = state.get("frequencies", {})
        vocab.frequencies.update(frequencies)
        return vocab

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        *,
        tokenizer=None,
        min_frequency: int = 1,
        specials: Iterable[str] | None = None,
        unk_token: str | None = "<unk>",
    ) -> "Vocab":
        if min_frequency < 1:
            raise ValueError("min_frequency must be at least 1")

        if tokenizer is None:
            tokenizer = str.split

        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(tokenizer(text))

        sorted_tokens = sorted(
            (token for token, count in counter.items() if count >= min_frequency),
            key=lambda token: (-counter[token], token),
        )

        vocab = cls(specials=specials, unk_token=unk_token)
        for token in sorted_tokens:
            vocab._add_token(token, count_frequency=False)
        vocab.frequencies.update(counter)
        return vocab

    @property
    def token_to_index(self) -> dict[str, int]:
        return dict(self._token_to_index)

    @property
    def index_to_token(self) -> list[str]:
        return list(self._index_to_token)
