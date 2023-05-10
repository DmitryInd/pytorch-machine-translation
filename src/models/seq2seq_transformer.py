import torch
import torch.nn as nn

from models.positional_encoding import PositionalEncoding

import metrics


class Seq2SeqTransformer(torch.nn.Module):
    def __init__(self, device, encoder_vocab_size, decoder_vocab_size, target_tokenizer, start_symbol, lr, total_steps,
                 emb_size=512,
                 num_heads=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dropout=0.1,
                 div_factor=10e+4):
        super(Seq2SeqTransformer, self).__init__()
        self.device = device
        self.max_sent_len = target_tokenizer.max_sent_len
        self.target_tokenizer = target_tokenizer
        self.start_id = self.target_tokenizer.word2index[start_symbol]
        self.encoder_embedding = nn.Embedding(encoder_vocab_size, emb_size).to(self.device)
        self.decoder_embedding = nn.Embedding(decoder_vocab_size, emb_size).to(self.device)
        self.positional_encoder = PositionalEncoding(
            emb_size=emb_size, max_len=self.max_sent_len
        ).to(self.device)
        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout
        ).to(self.device)
        self.voc_proj = nn.Linear(emb_size, decoder_vocab_size).to(self.device)
        # Optimization parameters
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=self.optimizer,
            total_steps=total_steps,
            max_lr=lr,
            pct_start=0.1,
            anneal_strategy='linear',
            final_div_factor=div_factor
        )

    def generate_square_subsequent_mask(self, length: int):
        mask = torch.tril(torch.ones((length, length), device=self.device)).float()
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0))
        return mask

    def forward(self, input_tensor: torch.Tensor):
        # input_tensor: (B, S), S - sequence length, B - batch_size
        # Embedding + positional encoding: (S, B, E), E - encoder_embedding size
        src = self.positional_encoder(self.encoder_embedding(input_tensor.transpose(0, 1)))
        # A memory is an encoder output: (S, B, E):
        memory = self.transformer.encoder(src)
        # Output
        pred_tokens = [torch.full((input_tensor.size(0),), self.start_id)]
        each_step_distributions = [nn.functional.one_hot(pred_tokens[0],
                                                         self.voc_proj.out_features).to(self.device).float()]
        each_step_distributions[0] = each_step_distributions[0].masked_fill(
            each_step_distributions[0] == 0, float('-inf')
        ).masked_fill(
            each_step_distributions[0] == 1, float(0)
        )
        # (S, B), where S is the length of the predicted sequence
        prediction = torch.full((1, input_tensor.size(0)), self.start_id, dtype=torch.long, device=self.device)
        for i in range(self.max_sent_len - 1):
            tgt_mask = self.generate_square_subsequent_mask(prediction.size(0))
            out = self.transformer.decoder(self.decoder_embedding(prediction), memory, tgt_mask)
            logits = self.voc_proj(out[-1])
            _, next_word = torch.max(logits, dim=1)
            prediction = torch.cat([prediction, next_word.unsqueeze(0)], dim=0)
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
        self.scheduler.step()

        return loss.item()

    @torch.no_grad()
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
