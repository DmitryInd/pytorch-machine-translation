import re
from typing import List, Tuple

from tqdm import tqdm


class TextUtils:
    @staticmethod
    def normalize_text(s: str) -> str:
        """Normalizes string, removes punctuation and
        non alphabet symbols

        Args:
            s (str): string to normalize

        Returns:
            str: normalized string
        """
        s = s.lower()
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Zа-яйёьъА-ЯЙ]+", r" ", s)
        s = s.strip()
        return s

    @staticmethod
    def read_langs_pairs_from_file(filename: str):
        """Read lang from file

        Args:
            filename (str): path to dataset

        Returns:
            List[Tuple[str, str]]: string pairs
        """
        with open(filename, mode="r", encoding="utf-8") as f:
            lines = f.read().strip().split("\n")

        lang_pairs = []
        for line in tqdm(lines, desc="Reading from file"):
            lang_pair = tuple(map(TextUtils.normalize_text, line.split("\t")[:2]))
            lang_pairs.append(lang_pair)

        return lang_pairs


def short_text_filter_function(x, max_length, prefix_filter=None):
    def len_filter(x_in):
        return len(x_in[0].split(" ")) <= max_length and len(x_in[1].split(" ")) <= max_length

    if prefix_filter:
        def prefix_filter_func(x_in):
            return x_in[0].startswith(prefix_filter)
    else:
        def prefix_filter_func(_):
            return True
    return len_filter(x) and prefix_filter_func(x)
