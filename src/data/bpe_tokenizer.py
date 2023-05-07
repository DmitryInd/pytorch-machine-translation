from typing import List

from tokenizers import Tokenizer
from tokenizers import decoders
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer


class BPETokenizer:
    def __init__(self, sentence_list, pad_flag):
        """
        sentence_list - список предложений для обучения
        """
        self._special_tokens_set = {"[SOS]", "[EOS]", "[PAD]"}
        self._pad_token = "[PAD]"

        self._tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        trainer = BpeTrainer(special_tokens=["[UNK]", "[SOS]", "[EOS]", "[PAD]"])

        self._tokenizer.pre_tokenizer = Whitespace()
        self._tokenizer.decoder = decoders.BPEDecoder()
        self._tokenizer.train_from_iterator(sentence_list, trainer)

        self._tokenizer.post_processor = TemplateProcessing(
            single="[SOS] $A [EOS]",
            pair="[SOS] $A [EOS] $B:1 [EOS]:1",
            special_tokens=[
                ("[SOS]", self._tokenizer.token_to_id("[SOS]")),
                ("[EOS]", self._tokenizer.token_to_id("[EOS]")),
            ]
        )

        if pad_flag:
            max_length = self._get_max_length_in_tokens(sentence_list)
            self._tokenizer.enable_padding(pad_id=self._tokenizer.token_to_id(self._pad_token),
                                           length=max_length)
            self._tokenizer.enable_truncation(max_length=max_length)

    def __call__(self, sentence):
        """
        sentence - входное предложение
        """
        return self._tokenizer.encode(sentence).ids

    def decode(self, token_list):
        """
        token_list - предсказанные ID вашего токенизатора
        """
        # predicted_tokens = self._tokenizer.decode(token_list)
        predicted_tokens = []
        for token_id in token_list:
            predicted_token = self._tokenizer.id_to_token(token_id)
            predicted_tokens.append(predicted_token)
        predicted_tokens = list(filter(lambda x: x not in self._special_tokens_set, predicted_tokens))
        return predicted_tokens

    def _get_max_length_in_tokens(self, sentence_list: List[str]) -> int:
        max_length = 0
        for encoded in self._tokenizer.encode_batch(sentence_list):
            max_length = max(max_length, len(encoded.ids))
        return max_length
