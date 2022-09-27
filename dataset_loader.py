from lib2to3.pgen2 import token
from transformers import AutoTokenizer
import datasets
import numpy as np
from small_text.integrations.transformers.datasets import TransformersDataset


def load_my_dataset(dataset: str, transformer_model_name: str, tokenization=True):
    if dataset == "ag_news":
        # works
        raw_dataset = datasets.load_dataset("ag_news")
    elif dataset == "trec6":
        # works
        raw_dataset = datasets.load_dataset("trec")
        raw_dataset = raw_dataset.rename_column("label-coarse", "label")
    elif dataset == "subj":
        # works
        raw_dataset = datasets.load_dataset("SetFit/subj")
    elif dataset == "rotten":
        # works
        raw_dataset = datasets.load_dataset("rotten_tomatoes")
    elif dataset == "imdb":
        # works
        raw_dataset = datasets.load_dataset("imdb")
    elif dataset == "sst2":
        raw_dataset = datasets.load_dataset("sst2")
    elif dataset == "cola":
        raw_dataset = datasets.load_dataset("glue", "cola")
    else:
        print("dataset not known")
        exit(-1)

    print("First 3 training samples:\n")
    for i in range(3):
        print(raw_dataset["train"]["label"][i], " ", raw_dataset["train"]["text"][i])

    num_classes = np.unique(raw_dataset["train"]["label"]).shape[0]

    if tokenization:
        tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)

        def _get_transformers_dataset(tokenizer, data, labels, max_length=60):

            data_out = []

            for i, doc in enumerate(data):
                encoded_dict = tokenizer.encode_plus(
                    doc,
                    add_special_tokens=True,
                    padding="max_length",
                    max_length=max_length,
                    return_attention_mask=True,
                    return_tensors="pt",
                    truncation="longest_first",
                )

                data_out.append(
                    (
                        encoded_dict["input_ids"],
                        encoded_dict["attention_mask"],
                        labels[i],
                    )
                )

            return TransformersDataset(data_out)

        # print(raw_dataset['train']['text'][:10])
        # print(raw_dataset['train']['label'][:10])

        train = _get_transformers_dataset(
            tokenizer, raw_dataset["train"]["text"], raw_dataset["train"]["label"]
        )
        test = _get_transformers_dataset(
            tokenizer, raw_dataset["test"]["text"], raw_dataset["test"]["label"]
        )

        return train, test, num_classes
    else:
        return raw_dataset["train"], raw_dataset["test"], num_classes
