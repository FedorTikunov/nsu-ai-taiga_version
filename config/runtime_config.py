import json
import sys
import os

if "ONLYFANS_CFG" in os.environ:
    print(f"Using env config: {os.environ['ONLYFANS_CFG']}")
    with open(os.environ["ONLYFANS_CFG"]) as f:
        config = json.load(f)["runtime"]
    d = locals()
    for k, v in config.items():
        # setattr(sys.modules[__name__], k, v)
        d[k] = v
else:
    print("Using default config")
    promt_prefix = "<s>[INST]"
    initial_promt = ('You are a useful and friendly assistant with great erudition and '
                        'developed intelligence. You can keep up a conversation on various topics and even know '
                        'how to play complex intellectual games. ')
    image_text = "I have just looked at this <image>."
    image_caption_prefix = "I think image has this caption: "
    wiki_text_prefix = "I can describe this image as: "
    ocr_prefix = "Image has this text: "
    yolo_prefix = "I think image has objects: "
    yolo_captions_prefix = ""
    wiki_yolo_texts_prefix = "with description: "
    audio_text_prefix = "There is audio with this text: "

    answer_postfix = "[/INST]"
    promt_postfix = ""

    bot_use_history = False

    use_one_peace = True
    use_blit = True
    use_ocr = True
    use_yolo = True
    use_translation = True

    annoy_search_k = -1
    annoy_include_n_texts = 2

    annoy_input_weight = 0.1
    annoy_image_weight = 0.2
    annoy_caption_weight = 0.7
    annoy_ocr_weight = 0
    annoy_yolo_caption_weight = 0.1
    annoy_yolo_image_weight = 0.1
    
    annoy_audio_weight = 0.5

    yolo_use_blip_caption = True

    yolo_include_one_peace = True
    yolo_annoy_search_k = -1
    yolo_annoy_include_n_texts = 1
    yolo_one_peace_caption_weight = 0.5
    yolo_one_peace_image_weight = 0.5

