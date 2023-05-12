import os

import torch
import yaml
from models import trainer
from data.datamodule import DataManager
from txt_logger import TXTLogger
from models.seq2seq_transformer import Seq2SeqTransformer

if __name__ == "__main__":
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    if torch.cuda.is_available():
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'

    data_config = yaml.load(open("configs/data_config.yaml", 'r'),   Loader=yaml.Loader)
    dm = DataManager(data_config, DEVICE)
    train_dataloader, dev_dataloader = dm.prepare_data()

    model_config = yaml.load(open("configs/transformer_config.yaml", 'r'),   Loader=yaml.Loader)

    model = Seq2SeqTransformer(
        device=DEVICE,
        encoder_vocab_size=len(dm.source_tokenizer.index2word),
        decoder_vocab_size=len(dm.target_tokenizer.index2word),
        target_tokenizer=dm.target_tokenizer,
        start_symbol=dm.target_tokenizer.sos_token,
        lr=model_config['learning_rate'],
        total_steps=model_config['epoch_num']*len(train_dataloader),
        emb_size=model_config['emb_size'],
        num_heads=model_config['num_heads'],
        num_encoder_layers=model_config['num_encoder_layers'],
        num_decoder_layers=model_config['num_decoder_layers'],
        dropout=model_config['dropout'],
        div_factor=model_config['div_factor']
    )

    logger = TXTLogger('training_logs')
    trainer_cls = trainer.Trainer(model=model, model_config=model_config, logger=logger)

    if model_config['try_one_batch']:
        train_dataloader = [list(train_dataloader)[0]]
        dev_dataloader = [list(train_dataloader)[0]]

    trainer_cls.train(train_dataloader, dev_dataloader)




