from team_code.generate import generate_full_prompt, generate_answer_based_on_prompt, load_images, MultimodalModel, DEVICE
import json
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoModelForCausalLM
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

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
        human_texts = [item['human_text'] for item in samples]
        gpt_texts = [item['gpt_text'] for item in samples]

        loaded_images= load_images(images)
        
        p_inputs = []
        for prompt, image in zip(human_texts, loaded_images):
            p_inputs.append(self.processor(text=prompt, images=image, return_tensors="pt"))

        return {'input': p_inputs, 'label': gpt_texts}

def find_all_linear_names(model):
    """
    :param model: модель
    :return: list: список линейных слоев, оставленных для тренировки
    """
    cls = torch.nn.Linear  # Use the standard Linear class for LLaVA-NeXT
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        if 'lm_head' in lora_module_names:  # Remove 'lm_head' if it's not applicable for LLaVA-NeXT
            lora_module_names.remove('lm_head')
    return list(lora_module_names)


def train(model: MultimodalModel, processor: LlavaNextProcessor, batch_size: int, epochs: int, dir_train_dataset='train_dataset.json', dir_eval_dataset='eval_dataset.json'):
    

    modules = find_all_linear_names(model.llm) 
    train_dataset = Llava_finetuning_Dataset(json_file=dir_train_dataset, processor=processor)

    eval_dataset = Llava_finetuning_Dataset(json_file=dir_eval_dataset, processor=processor)
    collator = Collator(processor)

    # Configuration
    lora_config = LoraConfig( 
        r=8,
        lora_alpha=32,
        target_modules=modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    llm_model = get_peft_model(model.llm, lora_config)
    llm_model


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
        model=llm_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        device = DEVICE,
    )
    # Train the model
    trainer.train()