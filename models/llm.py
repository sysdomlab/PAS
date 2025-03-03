
from enum import Enum
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, TaskType, LoraConfig

class DefaultToken(Enum):
    PAD_TOKEN = "[PAD]"
    EOS_TOKEN = "</s>"
    BOS_TOKEN = "<s>"
    UNK_TOKEN = "<unk>"
    IGNORE_INDEX = -100

def get_model_from_huggingface(model_name, config):
    """
    Load a causal language model from HuggingFace transformers library.

    Args:
        model_name (str): The name of the pre-trained model to load.
        config (Config): The configuration object that contains the model
            parameters.

    Returns:
        AutoModelForCausalLM: A causal language model object.
    """


    return AutoModelForCausalLM.from_pretrained(model_name, **config)


def get_llm(model_name, llm_config, adapter_config):
    model = AutoModelForCausalLM.from_pretrained(model_name, **llm_config)

    # tokenizer = AutoTokenizer.from_pretrained(model_name,)
    # if num_new_tokens > 0:
    #     input_embeddings = model.get_input_embeddings().weight.data
    #     output_embeddings = model.get_output_embeddings().weight.data

    #     input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
    #         dim=0, keepdim=True)
    #     output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
    #         dim=0, keepdim=True)

    #     input_embeddings[-num_new_tokens:] = input_embeddings_avg
    #     output_embeddings[-num_new_tokens:] = output_embeddings_avg

    lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, **adapter_config)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def get_tokenizer(model_name, tok_len=128):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=tok_len,
        padding_side="right",
        use_fast=False,
    )

    special_tokens = dict()
    if tokenizer.pad_token is None:
        special_tokens["pad_token"] = DefaultToken.PAD_TOKEN.value
    if tokenizer.eos_token is None:
        special_tokens["eos_token"] = DefaultToken.EOS_TOKEN.value
    if tokenizer.bos_token is None:
        special_tokens["bos_token"] = DefaultToken.BOS_TOKEN.value
    if tokenizer.unk_token is None:
        special_tokens["unk_token"] = DefaultToken.UNK_TOKEN.value

    num_new_tokens = tokenizer.add_special_tokens(special_tokens)

    return tokenizer, num_new_tokens



if __name__ == "__main__":
    model_name = "/home/zgan/Models/llama-2-7b-hf"
    llm_config = {
    }
    adapter_config = {
        "r": 8,
        "lora_alpha": 8,
        "target_modules": ['q_proj', "v_proj"],
        "lora_dropout": 0.05,
    }
    model = get_llm(model_name, llm_config, adapter_config)

