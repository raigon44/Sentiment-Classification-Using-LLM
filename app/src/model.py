import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, pipeline
import numpy as np


class Model:

    def __init__(self, modelName, num_labels):
        self.tokenizer = AutoTokenizer.from_pretrained(modelName, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(modelName)
        self.modelName = modelName

    @staticmethod
    def compute_metrics(eval_preds):
        metric = evaluate.load("accuracy")
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    def fineTune(self, training_args, optimizer, scheduler, train_dataset, eval_dataset, test_dataset):

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            optimizers=(optimizer, scheduler),
            compute_metrics=self.compute_metrics,
        )

        trainer.train()

        train_loss = trainer.history["train_loss"]
        val_loss = trainer.history["eval_loss"]

        evaluation_results = trainer.evaluate(test_dataset)
        print(evaluation_results)

        return train_loss, val_loss

    def saveModel(self, location):
        self.model.save_pretrained(location+'/'+self.modelName+'fine_tuned.model')
        return

    def predict(self, input_text):
        classifier = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer)
        prediction = classifier(input_text)
        return prediction



