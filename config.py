import json
import sys
import os

if "ONLYFANS_CFG" in os.environ:
    with open(os.environ["ONLYFANS_CFG"]) as f:
        config = json.load(f)["runtime"]
    d = locals()
    for k, v in config.items():
        # setattr(sys.modules[__name__], k, v)
        d[k] = v
elif len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
    with open(sys.argv[1]) as f:
        config = json.load(f)["runtime"]
    d = locals()
    for k, v in config.items():
        # setattr(sys.modules[__name__], k, v)
        d[k] = v
else:
    initial_promt = ('<s>[INST] You are a useful and friendly assistant with great erudition and '
                        'developed intelligence. You can keep up a conversation on various topics and even know '
                        'how to play complex intellectual games. ')
    bot_use_history = True
    use_one_peace = True
    use_blit = True
    use_ocr = True
    max_wiki_paragraphs = 10
    annoy_search_k = -1
    use_annoy_dist = True
    annoy_input_weight = 0.1
    annoy_caption_weight = 0.7
    annoy_image_weight = 0.2
    annoy_ocr_weight = 0
    annoy_audio_weight = 0.5
    include_n_texts = 2
