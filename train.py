from team_code.generate import setup_model_and_tokenizer, load_images, MultimodalModel, DEVICE
import json
from transformers import LlavaNextProcessor
from torch.utils.data import Dataset
import torch
from transformers import Trainer, TrainingArguments
import sys

class Llava_finetuning_Dataset(Dataset):
    def __init__(self, json_file: str, processor: LlavaNextProcessor):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        item = self.data[idx]
        return {'text': item['conversations'][0]['value'], 'image': item['image'], 'answer': item['conversations'][1]['value']}

class Collator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, samples):

        images = [item['image'] for item in samples]
        human_texts = [item['text'] for item in samples]
        gpt_texts = [item['answer'] for item in samples]

        loaded_images= load_images(images)
        
        p_inputs = []
        for prompt, image in zip(human_texts, loaded_images):
            p_inputs.append(self.processor(text=prompt, images=image, return_tensors="pt"))

        return {'input': p_inputs, 'label': gpt_texts}


def train(model: MultimodalModel, processor: LlavaNextProcessor, batch_size: int, epochs: int, dir_train_dataset='train_dataset.json', dir_eval_dataset='eval_dataset.json'):
    

    train_dataset = Llava_finetuning_Dataset(json_file=dir_train_dataset, processor=processor)

    eval_dataset = Llava_finetuning_Dataset(json_file=dir_eval_dataset, processor=processor)
    collator = Collator(processor)

    # Configuration

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model.llm,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        device = DEVICE,
    )
    # Train the model
    trainer.train()

if __name__ == "__main__":

    model, processor = setup_model_and_tokenizer()
    train_json_path = sys.argv[1]
    eval_dataset = sys.argv[2]
    train(model, processor, 16, 100, train_json_path, eval_dataset)
