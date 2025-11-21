from datasets import Dataset, load_dataset 


def get_records_for_chatbot(dataset: Dataset):
    """
    Count records grouped by chatbot
    """
    return dataset.to_pandas()["chatbot"].value_counts()

dataset = load_dataset("daloopa/financial-retrieval", split="train")
print(get_records_for_chatbot(dataset))