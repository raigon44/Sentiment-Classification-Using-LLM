import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score


class Model:

    def __init__(self, modelName, num_labels):
        self.tokenizer = AutoTokenizer.from_pretrained(modelName, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(modelName)
        self.modelName = modelName

    def fineTune(self, training_args, optimizer, scheduler, train_dataset, eval_dataset, test_dataset):

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            optimizers=(optimizer, scheduler)
        )

        trainer.train()

        evaluation_results = trainer.evaluate(test_dataset)
        print(evaluation_results)

        return

    def saveModel(self, location):
        self.model.save_pretrained(location+'/'+self.modelName+'fine_tuned.model')
        return


