import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_openai import ChatOpenAI

def setup_llm(llm_model, llm_temperature, llm_max_tokens):
    llm_models = [
        "gpt-4o",
        "NCSOFT/Llama-VARCO-8B-Instruct",
        "rtzr/ko-gemma-2-9b-it",
        "mistralai/Mistral-Small-Instruct-2409"
    ]

    if llm_model in llm_models[1:]:
        model = AutoModelForCausalLM.from_pretrained(
            llm_model,
            device_map="balanced",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(llm_model)

        eos_token_id = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        text_generation_pipeline = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            temperature=llm_temperature,
            do_sample=True,
            return_full_text=False,
            max_new_tokens=llm_max_tokens,
            eos_token_id=eos_token_id,
        )

        llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
        torch.cuda.empty_cache()
        
    elif llm_model == llm_models[0]:
        llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o",
            temperature=llm_temperature,
            max_tokens=llm_max_tokens,
        )
    else:
        raise ValueError(f"지원하지 않는 llm_model입니다: {llm_model}")
    
    return llm