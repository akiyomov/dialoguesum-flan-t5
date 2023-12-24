from transformers import TrainingArguments, Trainer, logging
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
    print("Training started...")
    for epoch in range(trainer.args.num_train_epochs):
        print(f"Epoch {epoch + 1}/{trainer.args.num_train_epochs}:")
        for step, batch in enumerate(trainer.get_train_dataloader()):
            trainer.train()
            if step % trainer.args.logging_steps == 0:
                print(f"  Step {step}/{len(trainer.get_train_dataloader())}:")
                print(f"    Learning Rate: {trainer.lr_scheduler.get_lr()}")
                print(f"    Loss: {trainer.model.training_loss:.4f}")
                # Add printing of additional information here if needed
    print("Training finished.")

if __name__ == "__main__":
    model, tokenizer = load_pretrained_model('google/flan-t5-base')
    dataset = load_custom_dataset('knkarthick/dialogsum',tokenizer)

    training_args = TrainingArguments(
        output_dir='./dialogue-summary-training', 
        learning_rate=1e-5,
        num_train_epochs=5,
        per_device_train_batch_size=6,
        per_device_eval_batch_size=6,
        weight_decay=0.01,
        logging_dir='./logs', 
        report_to='tensorboard', 
    )
    logging.set_verbosity_info() 
    trainer = setup_trainer(model, training_args,dataset)

    train_model(trainer)
