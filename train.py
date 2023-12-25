from transformers import Trainer, TrainingArguments
import transformers
from torch.utils.tensorboard import SummaryWriter
from dataset import load_custom_dataset
from model import load_pretrained_model

def setup_trainer(model, training_args, dataset):
    tensorboard_writer = SummaryWriter(log_dir=training_args.logging_dir)
    tensorboard_callback = transformers.TrainerCallback(
        tensorboard_writer=tensorboard_writer,
        log_dir=training_args.logging_dir,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        callbacks=[tensorboard_callback],  # Add the TensorBoard callback here
    )
    return trainer

def train_model(trainer):
    print("Training started...")
    trainer.train()
    trainer.save_model()
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
        logging_steps=100,
        logging_dir='./logs', 
        report_to='tensorboard', 
    )
   
    trainer = setup_trainer(model, training_args,dataset)

    train_model(trainer)
