"""
Example run: python /userspace/pva/nsu-ai-taiga_version/bench.py {configs_dir / f'{not_evaluated_config_name}.json'} --batch-size 16 --sampled-root "/userspace/dra/updated_datasets" --answer-path "/userspace/pva/bench" --max 100 --step 5
"""
import team_code.generate as generate
import torch
import argparse
import datetime
from pathlib import Path
from torch.utils.data import Dataset
import json

from tqdm import tqdm
from torch.utils.data import DataLoader
import re
from typing import Optional

def remove_special_chars(s):
    pattern = r"[^a-zA-Z0-9\s]"
    s = re.sub(pattern, "", s)
    return s


def has_word(sentence, word):
    pattern = r"\b" + re.escape(word) + r"\b"
    match = re.search(pattern, sentence)
    if match:
        return True
    else:
        return False


class VQAEval:
    def __init__(self):
        self.contractions = {
            "aint": "ain't",
            "arent": "aren't",
            "cant": "can't",
            "couldve": "could've",
            "couldnt": "couldn't",
            "couldn'tve": "couldn't've",
            "couldnt've": "couldn't've",
            "didnt": "didn't",
            "doesnt": "doesn't",
            "dont": "don't",
            "hadnt": "hadn't",
            "hadnt've": "hadn't've",
            "hadn'tve": "hadn't've",
            "hasnt": "hasn't",
            "havent": "haven't",
            "hed": "he'd",
            "hed've": "he'd've",
            "he'dve": "he'd've",
            "hes": "he's",
            "howd": "how'd",
            "howll": "how'll",
            "hows": "how's",
            "Id've": "I'd've",
            "I'dve": "I'd've",
            "Im": "I'm",
            "Ive": "I've",
            "isnt": "isn't",
            "itd": "it'd",
            "itd've": "it'd've",
            "it'dve": "it'd've",
            "itll": "it'll",
            "let's": "let's",
            "maam": "ma'am",
            "mightnt": "mightn't",
            "mightnt've": "mightn't've",
            "mightn'tve": "mightn't've",
            "mightve": "might've",
            "mustnt": "mustn't",
            "mustve": "must've",
            "neednt": "needn't",
            "notve": "not've",
            "oclock": "o'clock",
            "oughtnt": "oughtn't",
            "ow's'at": "'ow's'at",
            "'ows'at": "'ow's'at",
            "'ow'sat": "'ow's'at",
            "shant": "shan't",
            "shed've": "she'd've",
            "she'dve": "she'd've",
            "she's": "she's",
            "shouldve": "should've",
            "shouldnt": "shouldn't",
            "shouldnt've": "shouldn't've",
            "shouldn'tve": "shouldn't've",
            "somebody'd": "somebodyd",
            "somebodyd've": "somebody'd've",
            "somebody'dve": "somebody'd've",
            "somebodyll": "somebody'll",
            "somebodys": "somebody's",
            "someoned": "someone'd",
            "someoned've": "someone'd've",
            "someone'dve": "someone'd've",
            "someonell": "someone'll",
            "someones": "someone's",
            "somethingd": "something'd",
            "somethingd've": "something'd've",
            "something'dve": "something'd've",
            "somethingll": "something'll",
            "thats": "that's",
            "thered": "there'd",
            "thered've": "there'd've",
            "there'dve": "there'd've",
            "therere": "there're",
            "theres": "there's",
            "theyd": "they'd",
            "theyd've": "they'd've",
            "they'dve": "they'd've",
            "theyll": "they'll",
            "theyre": "they're",
            "theyve": "they've",
            "twas": "'twas",
            "wasnt": "wasn't",
            "wed've": "we'd've",
            "we'dve": "we'd've",
            "weve": "we've",
            "werent": "weren't",
            "whatll": "what'll",
            "whatre": "what're",
            "whats": "what's",
            "whatve": "what've",
            "whens": "when's",
            "whered": "where'd",
            "wheres": "where's",
            "whereve": "where've",
            "whod": "who'd",
            "whod've": "who'd've",
            "who'dve": "who'd've",
            "wholl": "who'll",
            "whos": "who's",
            "whove": "who've",
            "whyll": "why'll",
            "whyre": "why're",
            "whys": "why's",
            "wont": "won't",
            "wouldve": "would've",
            "wouldnt": "wouldn't",
            "wouldnt've": "wouldn't've",
            "wouldn'tve": "wouldn't've",
            "yall": "y'all",
            "yall'll": "y'all'll",
            "y'allll": "y'all'll",
            "yall'd've": "y'all'd've",
            "y'alld've": "y'all'd've",
            "y'all'dve": "y'all'd've",
            "youd": "you'd",
            "youd've": "you'd've",
            "you'dve": "you'd've",
            "youll": "you'll",
            "youre": "you're",
            "youve": "you've",
        }
        self.manualMap = {
            "none": "0",
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
        }
        self.articles = ["a", "an", "the"]

        self.periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
        self.commaStrip = re.compile("(\d)(\,)(\d)")
        self.punct = [
            ";",
            r"/",
            "[",
            "]",
            '"',
            "{",
            "}",
            "(",
            ")",
            "=",
            "+",
            "\\",
            "_",
            "-",
            ">",
            "<",
            "@",
            "`",
            ",",
            "?",
            "!",
        ]

    def evaluate(self, answer, gt_answers):
        answer = answer.replace("\n", " ")
        answer = answer.replace("\t", " ")
        answer = answer.strip()
        answer = self.processPunctuation(answer)
        answer = self.processDigitArticle(answer)
        if type(gt_answers)==list:
            for i in range(len(gt_answers)):
                gt_answers[i] = gt_answers[i].replace("\n", " ")
                gt_answers[i] = gt_answers[i].replace("\t", " ")
                gt_answers[i] = gt_answers[i].strip()
                gt_answers[i] = self.processPunctuation(gt_answers[i])
                gt_answers[i] = self.processDigitArticle(gt_answers[i])
                if has_word(answer, gt_answers[i]):
                    return 1
            return 0
        else:
            gt_answers = gt_answers.replace("\n", " ")
            gt_answers= gt_answers.replace("\t", " ")
            gt_answers = gt_answers.strip()
            gt_answers = self.processPunctuation(gt_answers)
            gt_answers = self.processDigitArticle(gt_answers)
            if has_word(answer, gt_answers):
                return 1
            else:
                return 0
    
    def evaluate_MRR(self, answer, gt_answers):
        answer = answer.replace("\n", " ")
        answer = answer.replace("\t", " ")
        answer = answer.strip()
        answer = self.processPunctuation(answer)
        answer = self.processDigitArticle(answer)
        if type(gt_answers) is str:
            gt_answers = [gt_answers]
        for i in range(len(gt_answers)):
            gt_answers[i] = gt_answers[i].replace("\n", " ")
            gt_answers[i] = gt_answers[i].replace("\t", " ")
            gt_answers[i] = gt_answers[i].strip()
            gt_answers[i] = self.processPunctuation(gt_answers[i])
            gt_answers[i] = self.processDigitArticle(gt_answers[i])
            if has_word(answer, gt_answers[i]):
                return 1 / (i + 1)
        return 0.0

    def processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + " " in inText or " " + p in inText) or (
                re.search(self.commaStrip, inText) != None
            ):
                outText = outText.replace(p, "")
            else:
                outText = outText.replace(p, " ")
        outText = self.periodStrip.sub("", outText, re.UNICODE)
        return outText

    def processDigitArticle(self, inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = self.manualMap.setdefault(word, word)
            if word not in self.articles:
                outText.append(word)
            else:
                pass
        for wordId, word in enumerate(outText):
            if word in self.contractions:
                outText[wordId] = self.contractions[word]
        outText = " ".join(outText)
        return outText

class OnlyFansModel:

    def __init__(self):
        self.model, self.tokenizer = generate.setup_model_and_tokenizer()

    @torch.no_grad()
    def generate(self, image, question, max_new_tokens=64):
        if isinstance(image, Path):
            image = str(image.absolute())
        assert isinstance(image, str) 
        with torch.cuda.amp.autocast():
            cur_query_list = [{'type': 'image', 'content': image}, {'type': 'text', 'content': question}]
            response, history = generate.generate_text(self.model, self.tokenizer, cur_query_list=cur_query_list, history_list=("", ""))
        return response

    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens=1282):
        output = [self.generate(image, question, max_new_tokens) for image, question in zip(image_list, question_list)]
        return output

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("config", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sampled-root", type=Path, default=Path('./updated_datasets'))
    parser.add_argument("--answer-path", type=Path, default=Path("./tiny_answers"))
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--step", type=int, default=1)

    args = parser.parse_args()
    return args


class GeneralDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        root: Path = Path('./tiny_lvlm_datasets'),
        step: int = 1,
        max: Optional[int] = None,
    ):
        self.root = root
        self.dataset_name = dataset_name
        with open(root / dataset_name / "dataset.json", 'r') as f:
            if max is not None:
                self.dataset = json.load(f)[:max:step]
            else:
                self.dataset = json.load(f)[::step]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        sample['image_path'] = self.root / self.dataset_name / sample['image_path']
        return sample


def evaluate_VQA(
    model: OnlyFansModel,
    dataset: Dataset,
    model_name: str,
    dataset_name: str,
    task_type: str,
    time: datetime.datetime,
    batch_size: int = 1,
    answer_path: Path = Path('./answers')
):
    predictions=[]
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda batch: {key: [dict[key] for dict in batch] for key in batch[0]})
    eval = VQAEval()
    correct = 0
    num = 0
    for batch in tqdm(dataloader, desc="Running inference"):
        outputs = model.batch_generate(batch['image_path'], batch['question'])
        for image_path, question, gt_answer, output in zip(batch['image_path'], batch['question'], batch['gt_answers'], outputs):
            is_correct = eval.evaluate(output, gt_answer) == 1
            correct += is_correct
            num += 1
            answer_dict={
                'question': question,
                'answer': output,
                'gt_answers': gt_answer,
                'correct': is_correct,
                'image_path': str(image_path),
                'model_name': model_name,
                'task_type': task_type,
            }
            predictions.append(answer_dict)
    answer_file = answer_path / f"{dataset_name}.json"
    with open(answer_file, "w") as f:
        json.dump(predictions, f, indent=4)
    print(f'{dataset_name}:{float(correct)/num}')
    return float(correct)/num


def main(args):
    config_path: Path = args.config
    batch_size: int = args.batch_size
    sampled_root: Path = args.sampled_root
    answer_root: Path = args.answer_path
    max_samples = args.max
    step_smples = args.step
    bench_name = config_path.stem
    answer_dir: Path = answer_root / bench_name
    model = OnlyFansModel()
    time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    if not answer_dir.exists():
        answer_dir.mkdir()

    result = {}
    overall_score = 0
    dataset_names = ['Visual_Reasoning', 'Visual_Perception', 'Visual_Knowledge_Acquisition', 'Visual_Commonsense', 'Object_Hallucination']
    # Visual_Commonsense, Object_Hallucination
    for dataset_name in dataset_names:
        dataset = GeneralDataset(dataset_name, root=sampled_root, step=step_smples, max=max_samples)
        metrics = evaluate_VQA(model, dataset, bench_name, dataset_name, 'VQA', time, batch_size, answer_path=answer_dir)
        result[dataset_name] = metrics
        overall_score += metrics
    result['Overall_Score'] = overall_score
    print(f"Overall Score: {overall_score}")

    result_file = answer_dir / "result.json"
    with open(result_file, "w") as f:
        json.dump(result, f, indent=4)

if __name__ == "__main__":
    args = parse_args()
    main(args)
