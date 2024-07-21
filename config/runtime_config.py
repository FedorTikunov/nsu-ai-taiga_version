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
    initial_promt = ('<s>[INST] You are a useful and friendly assistant with great erudition and '
                        'developed intelligence. You can keep up a conversation on various topics and even know '
                        'how to play complex intellectual games. ')
    bot_use_history = False
    use_one_peace = True
    use_blit = True
    use_ocr = True
    use_yolo = True
    use_translation = True
    max_wiki_paragraphs = 10
    annoy_search_k = -1
    use_annoy_dist = True
    annoy_input_weight = 0.1
    annoy_caption_weight = 0.7
    annoy_image_weight = 0.2
    annoy_ocr_weight = 0
    annoy_audio_weight = 0.5
    include_n_texts = 2
    merge_yolo_objects = True
    yolo_objects_weight = 0.1
