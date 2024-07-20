import json
import sys
import os

if "ONLYFANS_CFG" in os.environ:
    with open(os.environ["ONLYFANS_CFG"]) as f:
        config = json.load(f)["startup"]
    d = locals()
    for k, v in config.items():
        # setattr(sys.modules[__name__], k, v)
        d[k] = v
else:
    weights_path = "/userspace/dra/nsu-ai/team_code/models"
    llava_weights = "/userspace/pva/weights/llava_next"
    debug = False
    flask_debug = False
    weights_ruen = "/userspace/pva/weights/opusruen"
    weights_enru = "/userspace/pva/weights/opusenru"
    weights_ocr = "/userspace/pva/weights/ocr"
    weights_whisper = "/userspace/pva/weights/whisper-medium"
    weights_yolo = "/userspace/pva/weights/yolov8/yolov8m.pt"
    weights_mistral = "/userspace/dra/nsu-ai/team_code/models/llm"
    weights_phi3 = "/userspace/dra/nsu-ai/team_code/models/Phi-3-mini-4k-instruct"
    weights_gemma2_9b = "/userspace/pva/weights/gemma2"
    load_one_peace = True
    load_blit = True
    load_ocr = True
    load_yolo = True
    load_sbert = True
    # llm_type = "llava"
    # llm_type = "mistral"
    # llm_type = "phi3"
    llm_type = "gemma2"
