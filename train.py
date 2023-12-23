from transformers import TrainingArguments, Trainer
from dataset import load_custom_dataset
from model import load_pretrained_model

def setup_trainer(model, training_args, dataset):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
    )
    return trainer

def train_model(trainer):
    trainer.train()

if __name__ == "__main__":
    # Load your model, tokenizer, and datasets here
    model, tokenizer = load_pretrained_model('google/flan-t5-base')
    dataset = load_custom_dataset('knkarthick/dialogsum',tokenizer)

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir='./dialogue-summary-training',  # Adjust this path as needed
        learning_rate=1e-5,
        num_train_epochs=5,
        per_device_train_batch_size=6,
        per_device_eval_batch_size=6,
        weight_decay=0.01,
    )

    # Setup Trainer
    trainer = setup_trainer(model, training_args,dataset)

    # Train the model
    train_model(trainer)
