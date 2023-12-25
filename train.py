from transformers import Trainer, TrainingArguments
import transformers
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainerCallback
from dataset import load_custom_dataset
from model import load_pretrained_model

class CustomTensorBoardCallback(TrainerCallback):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)

    def on_train_batch_end(self, args, state, control, logs, **kwargs):
        step = state.global_step
        self.writer.add_scalar("Loss/train", logs.get("loss"), step)
        self.writer.add_scalar("Learning_rate/train", logs.get("learning_rate"), step)


def setup_trainer(model, training_args, dataset):
    tensorboard_callback = CustomTensorBoardCallback(log_dir=training_args.logging_dir)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        callbacks=[tensorboard_callback],  # Add the CustomTensorBoardCallback here
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
