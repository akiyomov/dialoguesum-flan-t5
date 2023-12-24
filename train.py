from transformers import TrainingArguments, Trainer, logging
from dataset import load_custom_dataset
from model import load_pretrained_model

class LoggingCallback:
    def __init__(self, logging_steps=100):
        self.logging_steps = logging_steps

    def __call__(self, trainer):
        metrics = trainer.metrics
        if trainer.state.global_step % self.logging_steps == 0:
            logging.info(f"Training step {trainer.state.global_step}:")
            logging.info(f"  Learning Rate: {trainer.lr_scheduler.get_lr()}")
            logging.info(f"  Loss: {metrics['train']['loss']}")

def setup_trainer(model, training_args, dataset):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        callbacks=[LoggingCallback()]
    )
    return trainer

def train_model(trainer):
    print("Training started...")
    trainer.train()

if __name__ == "__main__":
    # Load your model, tokenizer, and datasets here
    model, tokenizer = load_pretrained_model('google/flan-t5-base')
    dataset = load_custom_dataset('knkarthick/dialogsum',tokenizer)

    training_args = TrainingArguments(
        output_dir='./dialogue-summary-training', 
        learning_rate=1e-5,
        num_train_epochs=5,
        per_device_train_batch_size=6,
        per_device_eval_batch_size=6,
        weight_decay=0.01,
        logging_steps=100, 
        evaluation_strategy='steps',
        eval_steps=100,  
        save_steps=500, 
        logging_dir='./logs', 
        report_to='tensorboard', 
    )

    logging.set_verbosity_info() 


    # Setup Trainer
    trainer = setup_trainer(model, training_args,dataset)

    # Train the model
    train_model(trainer)
