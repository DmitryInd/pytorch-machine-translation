import os

import torch
import yaml
from models import seq2seq_t5, trainer
from data.datamodule import DataManager
from txt_logger import TXTLogger

if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = 'cpu'

    data_config = yaml.load(open("configs/data_t5_config.yaml", 'r'),   Loader=yaml.Loader)
    dm = DataManager(data_config, DEVICE)
    train_dataloader, dev_dataloader = dm.prepare_data()

    model_config = yaml.load(open("configs/model_t5_config.yaml", 'r'),   Loader=yaml.Loader)

    model = seq2seq_t5.Seq2SeqT5(
        device=DEVICE,
        pretrained_name=model_config['pretrained_model_name'],
        encoder_vocab_size=len(dm.source_tokenizer.index2word),
        decoder_vocab_size=len(dm.target_tokenizer.index2word),
        target_tokenizer=dm.target_tokenizer,
        start_symbol=dm.target_tokenizer.pad_token,
        lr=model_config['learning_rate']
    )

    logger = TXTLogger('training_logs')
    trainer_cls = trainer.Trainer(model=model, model_config=model_config, logger=logger)

    if model_config['try_one_batch']:
        train_dataloader = [list(train_dataloader)[0]]
        dev_dataloader = [list(train_dataloader)[0]]

    trainer_cls.train(train_dataloader, dev_dataloader)




