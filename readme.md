# Baseline 

The algorithm code in a ZIP archive must be sent to the testing system. Solutions shall be run in an isolated environment using Docker

You can use an existing environment:     
From organizers - **cr.msk.sbercloud.ru/aijcontest_official/fbc3_0:0.1**   
From roman - **cr.msk.sbercloud.ru/aijcontest/roman:f1-0**   
Or your own docker image (example provided in the Data tab of competition)     

The link to the instruction of how to build and push custom Docker image is available in the Data tab.

Solution <ins>should</ins> be packed in a *team_code* folder with file **__init__.py**
and **generate.py** module with 3 main methods:
* **setup_model_and_tokenizer**() -> model, tokenizer
* **generate_text**(model: object, tokenizer: object, cur_query_list: List[Dict], history_list: Tuple[Object, str]) -> str, Object (history_on_current_round)
* **get_ppl**(model: object, tokenizer: object, cur_query_tuple: Tuple[List[Dict], str], history: Tuple[Object, str]) -> float, Object (history_on_current_round)

you can read more on a [competition page](https://dsworks.ru/en/champ/24cbf452-5c7b-44d4-99ba-b2e870742d23?#overview)



