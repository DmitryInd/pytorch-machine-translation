import torch
from torch.nn import CrossEntropyLoss, Embedding
from transformers import T5ForConditionalGeneration
from transformers.optimization import Adafactor

import metrics


class Seq2SeqT5(torch.nn.Module):
    def __init__(self, device,
                 pretrained_name: str,
                 encoder_vocab_size: int,
                 decoder_vocab_size: int,
                 target_tokenizer,
                 start_symbol,
                 lr,
                 is_source_target_equal=False):
        super(Seq2SeqT5, self).__init__()
        self.device = device
        self.max_sent_len = target_tokenizer.max_sent_len
        self.target_tokenizer = target_tokenizer
        self.start_id = self.target_tokenizer.word2index[start_symbol]
        self.model = T5ForConditionalGeneration.from_pretrained(pretrained_name).to(self.device)
        # Expanding the space of the encoder embeddings
        self.model.resize_token_embeddings(encoder_vocab_size)
        if not is_source_target_equal:
            # Replacing the target encoder for a new language
            new_embeddings = Embedding(decoder_vocab_size, self.model.config.d_model).to(self.device)
            self.model.decoder.set_input_embeddings(new_embeddings)
        # A new target decoder head for fine-tuning for a new task
        new_head = torch.nn.Linear(self.model.lm_head.in_features, decoder_vocab_size).to(self.device)
        self.model.set_output_embeddings(new_head)
        # Parameters of optimization
        self.criterion = CrossEntropyLoss().to(self.device)
        self.optimizer = Adafactor(self.model.parameters(),
                                   lr=lr,
                                   relative_step=False,
                                   warmup_init=False)

    def forward(self, input_tensor: torch.Tensor):
        # Output
        pred_tokens = []
        each_step_distributions = []
        # (B, S), where S is the length of the predicted sequence
        prediction = torch.full((input_tensor.size(0), 1), self.start_id, dtype=torch.long, device=self.device)
        memory = None  # Last hidden states of encoder
        for i in range(self.max_sent_len):
            model_output = self.model(
                input_ids=input_tensor,
                decoder_input_ids=prediction,
                encoder_outputs=memory,
                return_dict=True
            )
            memory = (model_output.encoder_last_hidden_state, )
            logits = model_output.logits.transpose(0, 1)[-1]
            _, next_word = torch.max(logits, dim=1)
            prediction = torch.cat([prediction, next_word.unsqueeze(1)], dim=1)
            # Output update
            pred_tokens.append(next_word.clone().detach().cpu())
            each_step_distributions.append(logits)

        return pred_tokens, each_step_distributions

    def training_step(self, batch):
        self.optimizer.zero_grad()
        input_tensor, target_tensor = batch  # (B, S)
        predicted, decoder_outputs = self.forward(input_tensor)
        target_tensor = target_tensor[:, :, None]
        target_length = target_tensor.shape[1]
        loss = 0.0
        for di in range(target_length):
            loss += self.criterion(
                decoder_outputs[di].squeeze(), target_tensor[:, di, :].squeeze()
            )
        loss = loss / target_length
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def validation_step(self, batch):
        input_tensor, target_tensor = batch
        predicted, decoder_outputs = self.forward(input_tensor)
        target_tensor = target_tensor[:, :, None]
        with torch.no_grad():
            target_length = target_tensor.shape[1]
            loss = 0
            for di in range(target_length):
                loss += self.criterion(
                    decoder_outputs[di].squeeze(), target_tensor[:, di, :].squeeze()
                )
            loss = loss / target_length

        return loss.item()

    def eval_bleu(self, predicted_ids_list, target_tensor):
        predicted = torch.stack(predicted_ids_list)
        predicted = predicted.squeeze().detach().cpu().numpy().swapaxes(0, 1)[:, 1:]
        actuals = target_tensor.squeeze().detach().cpu().numpy()[:, 1:]
        bleu_score, actual_sentences, predicted_sentences = metrics.bleu_scorer(
            predicted=predicted, actual=actuals, target_tokenizer=self.target_tokenizer
        )
        return bleu_score, actual_sentences, predicted_sentences
