from datasets import Dataset, load_dataset 


def get_records_for_chatbot(dataset: Dataset):
    """
    Count records grouped by chatbot
    """
    return dataset.to_pandas()["chatbot"].value_counts()

def get_error_rate(dataset: Dataset):
    """
    Computes the error rate based on is_correct column
    """

    return dataset.to_pandas().groupby("chatbot")["is_correct"].mean()

if __name__ == "__main__":
    dataset = load_dataset("daloopa/financial-retrieval", split="train")
    print(get_records_for_chatbot(dataset))
    print(get_error_rate(dataset))
