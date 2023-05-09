from typing import List

from tokenizers import Tokenizer
from tokenizers import decoders
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer
from transformers import T5Tokenizer, PreTrainedTokenizer


class BPETokenizer:
    def __init__(self, sentence_list, pad_flag, pretrained_name: str = None):
        """
        sentence_list - список предложений для обучения
        """
        # Initialisation
        self.max_sent_len = None
        self.pad_flag = pad_flag
        if pretrained_name is None:
            self._tokenizer = self._train(sentence_list)
        else:
            self._tokenizer = self._load(pretrained_name, sentence_list)

        # Preparing dictionaries mapping tokens and ids
        self.word2index = self._tokenizer.get_vocab()
        self.index2word = {w_id: word for word, w_id in self.word2index.items()}

    def __call__(self, sentence, pretrained_force_padding: bool = None):
        """
        sentence - входное предложение
        """
        if self._pretrained:
            padding = self.pad_flag if pretrained_force_padding is None else pretrained_force_padding
            id_list = self._tokenizer.encode(sentence,
                                             padding='max_length' if padding else False,
                                             truncation=padding,
                                             max_length=self.max_sent_len)
        else:
            id_list = self._tokenizer.encode(sentence).ids
        return id_list

    def decode(self, token_list):
        """
        token_list - предсказанные ID вашего токенизатора
        """
        predicted_tokens = self._tokenizer.decode(token_list, skip_special_tokens=True).split()
        return predicted_tokens

    def _get_max_length_in_tokens(self, sentence_list: List[str]) -> int:
        max_length = 0
        for sentence in sentence_list:
            max_length = max(max_length, len(self(sentence, False)))
        return max_length

    def _train(self, sentence_list: List[str]) -> Tokenizer:
        # Pretrained flag
        self._pretrained = False
        # Special tokens
        self.unknown_token = "[UNK]"
        self.sos_token = "[SOS]"
        self.eos_token = "[EOS]"
        self.pad_token = "[PAD]"
        # Initialization
        self._tokenizer = Tokenizer(BPE(unk_token=self.unknown_token))
        self._tokenizer.pre_tokenizer = Whitespace()
        self._tokenizer.decoder = decoders.BPEDecoder()
        # Training
        trainer = BpeTrainer(special_tokens=[self.unknown_token, self.sos_token, self.eos_token, self.pad_token],
                             end_of_word_suffix="</w>")
        self._tokenizer.train_from_iterator(sentence_list, trainer)
        # Post-processing
        self._tokenizer.post_processor = TemplateProcessing(
            single="[SOS] $A [EOS]",
            pair="[SOS] $A [EOS] $B:1 [EOS]:1",
            special_tokens=[
                ("[SOS]", self._tokenizer.token_to_id(self.sos_token)),
                ("[EOS]", self._tokenizer.token_to_id(self.eos_token)),
            ]
        )
        if self.pad_flag:
            self.max_sent_len = self._get_max_length_in_tokens(sentence_list)
            self._tokenizer.enable_padding(pad_id=self._tokenizer.token_to_id(self.pad_token),
                                           length=self.max_sent_len)
            self._tokenizer.enable_truncation(max_length=self.max_sent_len)
        return self._tokenizer

    def _load(self, pretrained_name: str, sentence_list) -> PreTrainedTokenizer:
        # Pretrained flag
        self._pretrained = True
        # Download
        self._tokenizer = T5Tokenizer.from_pretrained(pretrained_name)
        if self.pad_flag:
            self.max_sent_len = self._get_max_length_in_tokens(sentence_list)
        # Special tokens
        self.unknown_token = self._tokenizer.unk_token
        self.sos_token = self._tokenizer.pad_token
        self.eos_token = self._tokenizer.eos_token
        self.pad_token = self._tokenizer.pad_token
        return self._tokenizer
