import torch
import torch.nn.functional as F

from imagebind import data
from imagebind.models.imagebind_model import ModalityType

DEVICE = "cuda:0"
EMB_DIM = 4096
N_MODALITY_EMBS = 32


# utils function that parses the format of the input query to a single sequence
def get_query_from_input(model, tokenizer, input_list):
    base_model = model[0]
    model_imagebind = model[1]
    img_tokens_emb = model[2]
    audio_tokens_emb = model[3]
    projection = model[4]

    all_emb = []

    ai_ids = tokenizer.encode("\n Bot: ", add_special_tokens=False, return_tensors="pt").to(DEVICE)
    ai_embeddings = base_model.model.embed_tokens(ai_ids)

    prompt = "\nUser: "
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(DEVICE)
    prompt_embeddings = base_model.model.embed_tokens(prompt_ids)
    all_emb.append(prompt_embeddings)

    for el in input_list:
        if el["type"] == "text":
            query = el["content"]
            query_ids = tokenizer.encode(query, add_special_tokens=False, return_tensors="pt").to(DEVICE)
            query_embeddings = base_model.model.embed_tokens(query_ids)
            all_emb.append(query_embeddings)
        elif el["type"] == "image":
            modality_start_emb, modality_end_emb = img_tokens_emb
            filepath = f"{el['content']}"
            if filepath in model_imagebind.query_dict:
                projected_modality_embs = model_imagebind.query_dict[filepath]
            else:
                modality_embedding = encode_image(model_imagebind, filepath).to(device=base_model.device, dtype=base_model.dtype)
                projected_modality_embs = projection(modality_embedding).to(device=base_model.device, dtype=base_model.dtype)
                model_imagebind.query_dict[filepath] = projected_modality_embs
            all_emb.extend(
                [
                    modality_start_emb[None, None].to(device=base_model.device, dtype=base_model.dtype),
                    projected_modality_embs.reshape(1, N_MODALITY_EMBS, EMB_DIM), 
                    modality_end_emb[None, None].to(device=base_model.device, dtype=base_model.dtype),
                ]
            )
        else:
            modality_start_emb, modality_end_emb = audio_tokens_emb
            filepath = f"{el['content']}"
            if filepath in model_imagebind.query_dict:
                projected_modality_embs = model_imagebind.query_dict[filepath]
            else:
                modality_embedding = encode_audio(model_imagebind, filepath).to(device=base_model.device, dtype=base_model.dtype)
                projected_modality_embs = projection(modality_embedding).to(device=base_model.device, dtype=base_model.dtype)
                model_imagebind.query_dict[filepath] = projected_modality_embs
            all_emb.extend(
                [
                    modality_start_emb[None, None].to(device=base_model.device, dtype=base_model.dtype),
                    projected_modality_embs.reshape(1, N_MODALITY_EMBS, EMB_DIM), 
                    modality_end_emb[None, None].to(device=base_model.device, dtype=base_model.dtype),
                ]
            )

        all_emb.append(ai_embeddings)

        embeddings = torch.cat(
            all_emb,
            dim=1,
        )
    return embeddings


def get_text_emb(model, tokenizer, text):
    if text is None or len(text) == 0:
        text = "I don't know.\n"
    text_ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt").to(DEVICE)
    text_embeddings = model.model.embed_tokens(text_ids)
    return text_embeddings


@torch.no_grad()
def encode_audio(model_imagebind, audio_paths, normalize=True):
    if isinstance(audio_paths, str):
        audio_paths = [audio_paths]
    inputs = {
        ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths=audio_paths, device=DEVICE),
    }
    universal_embeddings = model_imagebind(inputs)[ModalityType.AUDIO].to(DEVICE)
    if normalize:
        universal_embeddings = F.normalize(universal_embeddings, dim=-1)
    return universal_embeddings


@torch.no_grad()
def encode_image(model_imagebind, image_paths, normalize=True):
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    inputs = {
        ModalityType.VISION: data.load_and_transform_vision_data(image_paths, DEVICE),
    }
    universal_embeddings = model_imagebind(inputs)[ModalityType.VISION].to(DEVICE)
    if normalize:
        universal_embeddings = F.normalize(universal_embeddings, dim=-1)
    return universal_embeddings
