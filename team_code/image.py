import logging
import os
from config.startup_config import DEVICE
from typing import List, Tuple
from PIL import Image
from team_code.model import MultimodalModel
from PIL.ImageFile import ImageFile
import torch
from config import runtime_config, startup_config
from torchvision import transforms
from ultralytics.engine.results import Results
import numpy as np
from torchvision.transforms import InterpolationMode
from team_code.text import find_long_text_similar_to_short_text


image_process_logger = logging.getLogger(__name__)


def generate_image_caption(rgb_opened_image: ImageFile, model: MultimodalModel) -> str:
    if DEVICE.type == "cpu":
        inputs = model.image[0](rgb_opened_image, return_tensors="pt").to(DEVICE)
    else:    
        inputs = model.image[0](rgb_opened_image, return_tensors="pt").to(DEVICE, torch.float16)
    out = model.image[1].generate(**inputs)
    output = model.image[0].decode(out[0], skip_special_tokens=True)
    return output


def extract_text_with_trocr(rgb_opened_image: ImageFile, model: MultimodalModel) -> str:
    
    # Process the image
    pixel_values = model.ocr[0](rgb_opened_image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(DEVICE)
    generated_ids = model.ocr[1].generate(pixel_values)

    # Generate text
    text = model.ocr[0].batch_decode(generated_ids, skip_special_tokens=True)[0]

    return text


def process_image(image_fname: str) -> torch.Tensor:
    if not os.path.isfile(image_fname):
        return None  # Return None if the image file does not exist
    
    # Open the image file
    if startup_config.llm_type == "llava":
        with Image.open(image_fname) as img:
            # Define the transformations: resize and tensor conversion
            transform = transforms.Compose([
                transforms.Resize((672, 672)),  # Resize to the size expected by LLaVA-NeXT
                transforms.ToTensor(),  # Convert the PIL Image to a tensor
            ])
            
            # Apply the transformations
            image_tensor = transform(img)
    else:
        image_tensor = transforms.PILToTensor()(Image.open(image_fname))
        
    return image_tensor


def detect_and_crop_objects(images: List[ImageFile], model: MultimodalModel) -> Tuple[List[List[Image.Image]], List[List[str]], List[List[float]]]:
    cropped_acc: List[List[Image.Image]] = []
    classes_acc: List[List[Image.Image]] = []
    probs_acc: List[List[Image.Image]] = []

    # Load image
    # image = Image.open(image_fname)
    for image in images:
        cropped_images: List[Image.Image] = []
        classes: List[Image.Image] = []
        probs: List[Image.Image] = []

        prediction: Results = model.yolo(image)[0]
        for box in prediction.boxes:
            # Get bounding box coordinates
            x_min, y_min, x_max, y_max = box.xyxy[0].tolist()

            # Crop object from image and append to list
            cropped_image = image.crop((x_min, y_min, x_max, y_max))
            cropped_images.append(cropped_image)
            classes.append(prediction.names[int(box.cls)])
            probs.append(float(box.conf))
        
        cropped_acc.append(cropped_images)
        classes_acc.append(classes)
        probs_acc.append(probs)

    return cropped_acc, classes_acc, probs_acc


def process_yolo_image_for_one_peace(image_list: List[ImageFile], device="cuda:0", return_image_sizes=False, dtype="fp16") -> List[torch.Tensor]:
    def cast_data_dtype(dtype, t):
        if dtype == "bf16":
            return t.to(dtype=torch.bfloat16)
        elif dtype == "fp16":
            return t.to(dtype=torch.half)
        else:
            return t

    CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)
    mean = CLIP_DEFAULT_MEAN
    std = CLIP_DEFAULT_STD
    transform = transforms.Compose([
        transforms.Resize(
            (256, 256),
            interpolation=InterpolationMode.BICUBIC
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    patch_images_list = []
    image_width_list = []
    image_height_list = []
    for f_image in image_list:
        image = f_image.convert("RGB")
        w, h = image.size
        patch_image = transform(image)
        patch_images_list.append(patch_image)
        image_width_list.append(w)
        image_height_list.append(h)
    src_images = torch.stack(patch_images_list, dim=0).to(device)
    src_images = cast_data_dtype(dtype, src_images)
    if return_image_sizes:
        image_widths = torch.tensor(image_width_list).to(device)
        image_heights = torch.tensor(image_height_list).to(device)
        return src_images, image_widths, image_heights
    else:
        return src_images


def load_images(image_file_list: List[str]) -> List[ImageFile]:
    
    if not image_file_list:
        return None
    return [Image.open(file).convert("RGB") for file in image_file_list]


def find_text_by_image(input_text: str, image_fname: str, model: MultimodalModel) -> str:
    if not os.path.isfile(image_fname):
        err_msg = f'The image "{image_fname}" does not exist!'
        image_process_logger.error(err_msg)
        raise ValueError(err_msg)
    if runtime_config.use_blit and startup_config.load_blit:
        image_caption = generate_image_caption(image_fname, model)
    else:
        image_caption = ''

    if runtime_config.use_ocr and startup_config.load_ocr:
        trocr_text = extract_text_with_trocr(image_fname, model)
        trocr_text = f'Image has such text: "{trocr_text}"'
    else:
        trocr_text = ''

    if runtime_config.use_yolo and startup_config.load_yolo:
        crop_image_list = detect_and_crop_objects(image_fname, model)
    else:
        crop_image_list = []

    if runtime_config.use_one_peace and startup_config.load_one_peace:
        if runtime_config.use_annoy_dist:
            vectors = []
            weights = []
            if image_caption:
                src_tokens = model.one_peace.process_text([image_caption])
                with torch.no_grad():
                    text_features = model.one_peace.extract_text_features(src_tokens)
                del src_tokens
                text_vector = model.pca.transform(text_features.cpu().type(torch.FloatTensor).numpy()[0:1])[0]
                del text_features
                vectors.append(text_vector)
                weights.append(runtime_config.annoy_caption_weight)
            if input_text:
                src_tokens = model.one_peace.process_text([input_text])
                with torch.no_grad():
                    text_features = model.one_peace.extract_text_features(src_tokens)
                del src_tokens
                text_vector = model.pca.transform(text_features.cpu().type(torch.FloatTensor).numpy()[0:1])[0]
                del text_features
                vectors.append(text_vector)
                weights.append(runtime_config.annoy_input_weight)
            if trocr_text:
                src_tokens = model.one_peace.process_text([trocr_text])
                with torch.no_grad():
                    text_features = model.one_peace.extract_text_features(src_tokens)
                del src_tokens
                text_vector = model.pca.transform(text_features.cpu().type(torch.FloatTensor).numpy()[0:1])[0]
                del text_features
                vectors.append(text_vector)
                weights.append(runtime_config.annoy_ocr_weight)
            src_images = model.one_peace.process_image([image_fname])
            with torch.no_grad():
                image_features = model.one_peace.extract_image_features(src_images)
            del src_images
            image_vector = model.pca.transform(image_features.cpu().type(torch.FloatTensor).numpy()[0:1])[0]
            del image_features
            vectors.append(image_vector)
            weights.append(runtime_config.annoy_image_weight)

            if runtime_config.use_yolo and startup_config.load_yolo and runtime_config.merge_yolo_objects:
                for crop_image in crop_image_list:
                    src_image = process_yolo_image_for_one_peace([crop_image], device=model.one_peace.device, dtype=model.one_peace.dtype)
                    with torch.no_grad():
                        image_features = model.one_peace.extract_image_features(src_image)
                        image_vector = model.pca.transform(image_features.cpu().type(torch.FloatTensor).numpy()[0:1])[0]
                        del image_features
                        vectors.append(image_vector)
                        weights.append(runtime_config.yolo_objects_weight)

            found_indices = model.annoy_index.get_nns_by_vector(np.average(vectors, axis=0, weights=weights), n=runtime_config.max_wiki_paragraphs, search_k=runtime_config.annoy_search_k)[:runtime_config.include_n_texts]
            del vectors
            del weights
            long_text = " ".join((model.texts[idx] for idx in found_indices))

            if runtime_config.use_yolo and startup_config.load_yolo and not runtime_config.merge_yolo_objects:
                yolo_texts = []
                for crop_image in crop_image_list:
                    src_image = process_yolo_image_for_one_peace([crop_image], device=model.one_peace.device, dtype=model.one_peace.dtype)
                    with torch.no_grad():
                        image_features = model.one_peace.extract_image_features(src_image)
                        image_vector = model.pca.transform(image_features.cpu().type(torch.FloatTensor).numpy()[0:1])[0]
                        del image_features
                        found_indices = model.annoy_index.get_nns_by_vector(image_vector, n=runtime_config.max_wiki_paragraphs, search_k=runtime_config.annoy_search_k)
                        yolo_texts.append(model.texts[found_indices[0]])
                        del found_indices
                long_text = long_text + " Image has objects with description: " + " ".join(yolo_texts)
        else:
            src_images = model.one_peace.process_image([image_fname])
            with torch.no_grad():
                image_features = model.one_peace.extract_image_features(src_images)
            del src_images
            image_vector = model.pca.transform(image_features.cpu().type(torch.FloatTensor).numpy()[0:1])[0]
            del image_features
            found_indices = model.annoy_index.get_nns_by_vector(image_vector, n=runtime_config.max_wiki_paragraphs, search_k=runtime_config.annoy_search_k)
            found_texts = [model.texts[idx] for idx in found_indices]
            del found_indices
            long_text = find_long_text_similar_to_short_text(image_caption, found_texts, model)
            if runtime_config.use_yolo and startup_config.load_yolo:
                for crop_image in crop_image_list:
                    src_image = process_yolo_image_for_one_peace([crop_image], device=model.one_peace.device, dtype=model.one_peace.dtype)
                    with torch.no_grad():
                        image_features = model.one_peace.extract_image_features(src_image)
                        image_vector = model.pca.transform(image_features.cpu().type(torch.FloatTensor).numpy()[0:1])[0]
                        del image_features
                        found_indices = model.annoy_index.get_nns_by_vector(image_vector, n=runtime_config.max_wiki_paragraphs, search_k=runtime_config.annoy_search_k)
                        found_texts = [model.texts[idx] for idx in found_indices]
                        del found_indices
                        long_text = long_text.strip() + " Aditional text: " + find_long_text_similar_to_short_text(image_caption, found_texts, model)
            # long_text = model.texts[found_indices[0]]
            del image_vector, found_indices
    else:
        long_text = ''

    result = ". ".join(filter(bool, (image_caption, long_text, trocr_text)))
    
    return result