import random
from datasets import Dataset, load_dataset 
from collections import defaultdict


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

def extract_random_questions(dataset: Dataset, sample_rate: float):
    if sample_rate > 1 or sample_rate < 0:
        raise ValueError(f"sample_rate must be between 0 and 1")

    buckets = defaultdict(list)
    categories = set(dataset['category'])
    for category in categories:
        subset = dataset.filter(lambda x: x['category'] == category)
        buckets[category].extend(subset.select(random.sample(range(len(subset)), int(sample_rate * len(subset)))))

    return buckets

if __name__ == "__main__":
    dataset = load_dataset("daloopa/financial-retrieval", split="train")
    print(get_records_for_chatbot(dataset))
    print(get_error_rate(dataset))

    questions = extract_random_questions(dataset, 0.1)
