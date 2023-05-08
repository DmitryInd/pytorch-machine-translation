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
        # Special tokens
        self.unknown_token = "[UNK]"
        self.sos_token = "[SOS]"
        self.eos_token = "[EOS]"
        self.pad_token = "[PAD]"
        self._special_tokens_list = [self.unknown_token, self.sos_token, self.eos_token, self.pad_token]
        # Initialisation
        self._tokenizer = Tokenizer(BPE(unk_token=self.unknown_token))
        self._tokenizer.pre_tokenizer = Whitespace()
        self._tokenizer.decoder = decoders.BPEDecoder()
        # Training
        trainer = BpeTrainer(special_tokens=self._special_tokens_list, end_of_word_suffix="</w>")
        self._tokenizer.train_from_iterator(sentence_list, trainer)
        self._tokenizer.post_processor = TemplateProcessing(
            single="[SOS] $A [EOS]",
            pair="[SOS] $A [EOS] $B:1 [EOS]:1",
            special_tokens=[
                ("[SOS]", self._tokenizer.token_to_id(self.sos_token)),
                ("[EOS]", self._tokenizer.token_to_id(self.eos_token)),
            ]
        )
        if pad_flag:
            self.max_sent_len = self._get_max_length_in_tokens(sentence_list)
            self._tokenizer.enable_padding(pad_id=self._tokenizer.token_to_id(self.pad_token),
                                           length=self.max_sent_len)
            self._tokenizer.enable_truncation(max_length=self.max_sent_len)
        # Preparing dictionaries mapping tokens and ids
        self.word2index = self._tokenizer.get_vocab()
        self.index2word = {w_id: word for word, w_id in self.word2index.items()}

    def __call__(self, sentence):
        """
        sentence - входное предложение
        """
        return self._tokenizer.encode(sentence).ids

    def decode(self, token_list):
        """
        token_list - предсказанные ID вашего токенизатора
        """
        predicted_tokens = self._tokenizer.decode(token_list).split()
        return predicted_tokens

    def _get_max_length_in_tokens(self, sentence_list: List[str]) -> int:
        max_length = 0
        for sentence in sentence_list:
            max_length = max(max_length, len(self._tokenizer.encode(sentence).ids))
        return max_length
