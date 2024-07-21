from dataclasses import dataclass
from typing import Tuple, List
import logging
import os
import pickle
import fileinput
import gc
from ultralytics import YOLO
from team_code.ONE_PEACE.one_peace.models.one_peace.hub_interface import OnePeaceHubInterface
from team_code.ONE_PEACE.one_peace.models import from_pretrained
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex
from sklearn.pipeline import Pipeline
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoFeatureExtractor, ASTForAudioClassification
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoModelForCausalLM
import torch

from config import startup_config

@dataclass
class MultimodalModel:
    image: Tuple[VisionEncoderDecoderModel, VisionEncoderDecoderModel]
    audio: Tuple[VisionEncoderDecoderModel, VisionEncoderDecoderModel]
    speech: VisionEncoderDecoderModel
    sbert: SentenceTransformer
    one_peace: OnePeaceHubInterface
    pca: Pipeline
    annoy_index: AnnoyIndex
    texts: List[str]
    ocr: TrOCRProcessor
    llm: LlavaNextForConditionalGeneration
    translate_ruen: pipeline
    translate_enru: pipeline
    yolo: YOLO
    

def setup_model_and_tokenizer() -> Tuple[MultimodalModel, AutoTokenizer]:
    model_init_logger = logging.getLogger("model_init")
    DEVICE = startup_config.DEVICE
    # if not torch.cuda.is_available():
    #     err_msg = 'CUDA is not available!'
    #     model_init_logger.error(err_msg)
    #     raise ValueError(err_msg)
    # if not os.path.isdir(one_peace_dir):
    #     err_msg = f'The directory "{one_peace_dir}" does not exist!'
    #     model_init_logger.error(err_msg)
    #     raise ValueError(err_msg)
    # sys.path.append(one_peace_dir)
    # from one_peace.models import from_pretrained

    # model_init_logger.info('ONE-PEACE is attached.')

    # model_dir = os.path.join(os.path.dirname(__file__), 'models')
    model_dir = startup_config.weights_path
    if not os.path.isdir(model_dir):
        err_msg = f'The directory "{model_dir}" does not exist!'
        model_init_logger.error(err_msg)
        raise ValueError(err_msg)

    one_peace_model_fname = os.path.join(model_dir, 'one-peace.pt')
    if not os.path.isfile(one_peace_model_fname):
        err_msg = f'The file "{one_peace_model_fname}" does not exist!'
        model_init_logger.error(err_msg)
        raise ValueError(err_msg)
    current_workdir = os.getcwd()
    model_init_logger.info(f'Current working directory: {current_workdir}')
    if startup_config.load_one_peace:
        one_peace_dir = os.path.join(os.path.dirname(__file__), 'ONE-PEACE')
        os.chdir(one_peace_dir)
        model_init_logger.info(f'New working directory: {os.getcwd()}')
        if DEVICE.type == "cpu":
            onepeace_model = from_pretrained(one_peace_model_fname, device=DEVICE)
        else:
            onepeace_model = from_pretrained(one_peace_model_fname, device=DEVICE, dtype='fp16')
        model_init_logger.info(f'ONE-PEACE model is loaded from the "{one_peace_model_fname}".')
        os.chdir(current_workdir)
        model_init_logger.info(f'Restored working directory: {os.getcwd()}')
        model_init_logger.info('The ONE-PEACE model is loaded.')

        pca_fname = os.path.join(model_dir, 'wiki_onepeace_pca.pkl')
        if not os.path.isfile(pca_fname):
            err_msg = f'The file "{pca_fname}" does not exist!'
            model_init_logger.error(err_msg)
            raise ValueError(err_msg)
        with open(pca_fname, 'rb') as fp:
            pca = pickle.load(fp)
        if not isinstance(pca, Pipeline):
            err_msg = (f'The PCA pipeline loaded from the "{pca_fname}" has a wrong type! '
                    f'Expected sklearn.pipeline.Pipeline, got {type(pca)}.')
            model_init_logger.error(err_msg)
            raise ValueError(err_msg)
        feature_vector_size = pca.named_steps.pca.n_components
        model_init_logger.info(f'The PCA pipeline is loaded. The feature vector size is {feature_vector_size}.')

        texts_fname = os.path.join(model_dir, 'en_wiki_paragraphs.txt')
        if not os.path.isfile(texts_fname):
            err_msg = f'The file "{texts_fname}" does not exist!'
            model_init_logger.error(err_msg)
            raise ValueError(err_msg)
        paragraphs = []
        counter = 0
        for curline in fileinput.input(texts_fname, openhook=fileinput.hook_encoded("utf-8", "surrogateescape")):
            prepline = curline.strip()
            if len(prepline) > 0:
                paragraphs.append(prepline)
                counter += 1
                if counter % 1_000_000 == 0:
                    model_init_logger.info(f'{counter} paragraphs are loaded from the "{texts_fname}".')
        gc.collect()
        info_msg = (f'The text corpus with Wikipedia paragraphs is loaded. '
                    f'There are {len(paragraphs)} paragraphs in this corpus.')
        model_init_logger.info(info_msg)

        annoy_fname = os.path.join(model_dir, 'en_wiki_paragraphs.ann')
        if not os.path.isfile(annoy_fname):
            err_msg = f'The file "{annoy_fname}" does not exist!'
            model_init_logger.error(err_msg)
            raise ValueError(err_msg)
        annoy_index = AnnoyIndex(feature_vector_size, 'angular')
        annoy_index.load(annoy_fname)
        model_init_logger.info('The Annoy index for Wikipedia paragraphs is loaded.')
        n_annoy_items = annoy_index.get_n_items()
        if n_annoy_items != len(paragraphs):
            err_msg = (f'The Wiki text corpus does not correspond to the Wiki text index, '
                    f'because their sizes are not same! {n_annoy_items} != {len(paragraphs)}.')
            model_init_logger.error(err_msg)
            raise ValueError(err_msg)
    else:
        onepeace_model = None
        pca = None
        annoy_index = None
        paragraphs = None

    audio_cls_dirname = os.path.join(model_dir, 'auxiliary_models', 'audioset')
    if not os.path.isdir(audio_cls_dirname):
        err_msg = f'The directory "{audio_cls_dirname}" does not exist!'
        model_init_logger.error(err_msg)
        raise ValueError(err_msg)
    audio_fe = AutoFeatureExtractor.from_pretrained(audio_cls_dirname)
    if DEVICE.type == "cpu":
        audio_cls = ASTForAudioClassification.from_pretrained(audio_cls_dirname).to(DEVICE)
    else:
        audio_cls = ASTForAudioClassification.from_pretrained(audio_cls_dirname, torch_dtype=torch.float16).to(DEVICE)
    
    if startup_config.load_blit:
        image_captioning_dirname = os.path.join(model_dir, 'auxiliary_models', 'blip')
        if not os.path.isdir(image_captioning_dirname):
            err_msg = f'The directory "{image_captioning_dirname}" does not exist!'
            model_init_logger.error(err_msg)
            raise ValueError(err_msg)
        image_processor = BlipProcessor.from_pretrained(image_captioning_dirname)
        if DEVICE.type == "cpu":
            image_caption_generator = BlipForConditionalGeneration.from_pretrained(
                image_captioning_dirname
            ).to(DEVICE)
        else:
            image_caption_generator = BlipForConditionalGeneration.from_pretrained(
                image_captioning_dirname,
                torch_dtype=torch.float16
            ).to(DEVICE)
    else:
        image_captioning_dirname = None
        image_processor = None
        image_caption_generator = None

    # asr_dirname = os.path.join(model_dir, 'auxiliary_models', 'whisper_medium')
    # if not os.path.isdir(asr_dirname):
    #     err_msg = f'The directory "{asr_dirname}" does not exist!'
    #     model_init_logger.error(err_msg)
    #     raise ValueError(err_msg)
    asr_pipe = pipeline(
        'automatic-speech-recognition',
        model=startup_config.weights_whisper,
        chunk_length_s=30,
        device=DEVICE
    )

    if startup_config.load_sbert:
        sbert_dirname = os.path.join(model_dir, 'auxiliary_models', 'sbert')
        if not os.path.isdir(sbert_dirname):
            err_msg = f'The directory "{sbert_dirname}" does not exist!'
            model_init_logger.error(err_msg)
            raise ValueError(err_msg)
        sentence_embedder = SentenceTransformer(sbert_dirname, device=DEVICE.type)
    else:
        sentence_embedder = None

    if startup_config.llm_type == "llava":
        llm_dirname = startup_config.llava_weights
        if not os.path.isdir(llm_dirname):
            err_msg = f'The directory "{llm_dirname}" does not exist!'
            model_init_logger.error(err_msg)
            raise ValueError(err_msg)

        if DEVICE.type == "cpu":
            llm_model = LlavaNextForConditionalGeneration.from_pretrained(llm_dirname).to(DEVICE)
        else:
            llm_model = LlavaNextForConditionalGeneration.from_pretrained(llm_dirname, torch_dtype=torch.float16, device_map={"":0})

        llm_model.eval()
        llm_processor= LlavaNextProcessor.from_pretrained(llm_dirname)
        model_init_logger.info('The large language model is loaded.')
    else:
        if startup_config.llm_type == "mistral":
            llm_dirname = startup_config.weights_mistral
        elif startup_config.llm_type == "phi3":
            llm_dirname = startup_config.weights_phi3
        elif startup_config.llm_type == "gemma2":
            llm_dirname = startup_config.weights_gemma2_9b
        else:
            raise ValueError('Unknown LLM type.')

        if DEVICE.type == "cpu":
            llm_model = AutoModelForCausalLM.from_pretrained(llm_dirname).to(DEVICE)
        else:
            llm_model = AutoModelForCausalLM.from_pretrained(llm_dirname, torch_dtype=torch.float16, device_map={"":0})

        llm_model.eval()
        llm_processor = AutoTokenizer.from_pretrained(llm_dirname)
        model_init_logger.info('The large language model is loaded.')

    # Load TrOCR model and processor
    if startup_config.load_ocr:
        trocr_processor = TrOCRProcessor.from_pretrained(startup_config.weights_ocr)
        if DEVICE.type == "cpu":
            trocr_model = VisionEncoderDecoderModel.from_pretrained(startup_config.weights_ocr).to(DEVICE)
        else:
            trocr_model = VisionEncoderDecoderModel.from_pretrained(startup_config.weights_ocr, torch_dtype=torch.float16).to(DEVICE)
        model_init_logger.info('The Ocr model is loaded.')
    else:
        trocr_processor = None
        trocr_model = None

    # Load YOLOv8     print(f"Using config: {os.environ['ONLYFANS_CFG']}")model and processor
    if startup_config.load_yolo:
        if DEVICE.type == "cpu":
            yolov8 = YOLO(startup_config.weights_yolo).to(DEVICE)
        else:
            yolov8 = YOLO(startup_config.weights_yolo).to(DEVICE)
        model_init_logger.info('The YOLOv8 model is loaded.')
    else:
        yolov8 = None

    translate_ruen = pipeline("translation", model=startup_config.weights_ruen, device=DEVICE)
    translate_enru = pipeline("translation", model=startup_config.weights_enru, device=DEVICE)
    model_init_logger.info('The Translation models are loaded.')

    full_pipeline_for_conversation = MultimodalModel(
        image=(image_processor, image_caption_generator),
        audio=(audio_fe, audio_cls),
        speech=asr_pipe,
        sbert=sentence_embedder,
        one_peace=onepeace_model,
        pca=pca,
        annoy_index=annoy_index,
        texts=paragraphs,
        ocr=(trocr_processor, trocr_model),
        llm=llm_model,
        translate_ruen=translate_ruen,
        translate_enru=translate_enru,
        yolo=yolov8,
    )
    gc.collect()
    return full_pipeline_for_conversation, llm_processor