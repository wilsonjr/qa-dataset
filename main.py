import random
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

def extract_random_questions(dataset: Dataset, n: int):
    if n > len(dataset):
        raise ValueError(f"n must be less than or equal to the length of the dataset ({len(dataset)})")
    if n <= 0:
        raise ValueError(f"n must be greater than 0")

    return dataset.select(random.sample(range(len(dataset)), n))['question']

if __name__ == "__main__":
    dataset = load_dataset("daloopa/financial-retrieval", split="train")
    print(get_records_for_chatbot(dataset))
    print(get_error_rate(dataset))

    questions = extract_random_questions(dataset, 50)
