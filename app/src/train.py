import data_utils
import configparser
from transformers import AdamW, TrainingArguments, get_linear_schedule_with_warmup
from model import Model
import pandas as pd


def main(pre_trained_model, dataset_name):

    config = configparser.ConfigParser()
    config.read('config.ini')
    train_df, val_df, test_df = data_utils.load_data(dataset_name)

    model_obj = Model(pre_trained_model, config.getint('IMDB', 'num_labels'))

    train_data_loader = data_utils.getDataLoader(train_df, model_obj.tokenizer, config.getint('Hyperparameter', 'per_device_train_batch_size'))
    validation_data_loader = data_utils.getDataLoader(val_df, model_obj.tokenizer, config.getint('Hyperparameter', 'per_device_eval_batch_size'))
    test_data_loader = data_utils.getDataLoader(test_df, model_obj.tokenizer, config.getint('Hyperparameter', 'per_device_eval_batch_size'))

    training_args = TrainingArguments(
        output_dir=config.get('Hyperparameter', 'output_dir'),
        num_train_epochs=config.getint('Hyperparameter', 'num_train_epochs'),
        per_device_train_batch_size=config.getint('Hyperparameter', 'per_device_train_batch_size'),
        per_device_eval_batch_size=config.getint('Hyperparameter', 'per_device_eval_batch_size'),
        save_steps=config.getfloat('Hyperparameter', 'save_steps'),
        save_total_limit=config.getint('Hyperparameter', 'save_total_limit'),
        evaluation_strategy=config.get('Hyperparameter', 'evaluation_strategy'),
        eval_steps=config.getint('Hyperparameter', 'eval_steps'),
        logging_steps=config.getint('Hyperparameter', 'logging_steps'),
        learning_rate=config.getfloat('Hyperparameter', 'learning_rate')
    )

    optimizer = AdamW(model_obj.model.parameters(), lr=config.getfloat('Hyperparameter', 'learning_rate'))

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_data_loader) * config.getint('Hyperparameter', 'num_train_epochs')
    )

    model_obj.fineTune(training_args, optimizer, scheduler, train_data_loader, validation_data_loader, test_data_loader)

    return


if __name__ == '__main__':
    main('bert-base-uncased', 'rotten_tomatoes')


