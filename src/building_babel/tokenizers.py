"""
This model will eventually use sentencepiece (I'm expecting we will want to use
llama's tokenizer), but for initial testing, here is a very simple character
based tokenizer.
"""

from sentencepiece import SentencePieceProcessor
from logging import getLogger
import os

logger = getLogger(__name__)


def codepoint_tokenize(s: str | list[str]):
    """This is a very simple unicode byte level tokenizer... essentially
    byte level bpe without merges.
    """
    if isinstance(s, str):
        return [int(c) for c in s.encode("utf-8")]
    else:
        ls = []
        for st in s:
            ls.append([int(c) for c in st.encode("utf-8")])  # noqa: PERF401
        return ls


def codepoint_decode(li: list[int] | list[list[int]]):
    if isinstance(li[0], int):
        return bytes(li).decode("utf-8") # type: ignore
    else:
        return [bytes(l).decode("utf-8") for l in li]


class LlamaTokenizer:
    def __init__(self, model_path: str):
        assert os.path.isfile(model_path), model_path

        self.sp_model = SentencePieceProcessor(model_file=model_path) # type: ignore
        logger.info(f"reloaded SentencePiece model from {model_path}")

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size() # type: ignore

    def encode(self, s: str, bos: bool, eos: bool) -> list[int]:
        assert type(s) is str
        t = self.sp_model.encode(s) # type: ignore
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: list[int]) -> str:
        return self.sp_model.decode(t) # type: ignore

    def __getitem__(self, key):
        return self.sp_model.IdToPiece(key)
