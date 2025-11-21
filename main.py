import random
import matplotlib.pyplot as plt
import seaborn as sns

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
    # print(get_records_for_chatbot(dataset))
    # questions = extract_random_questions(dataset, 0.1)

    
    error_rate = get_error_rate(dataset).reset_index()
    error_rate.columns = ['chatbot', 'error_rate']
    error_rate['error_rate'] = 1-error_rate['error_rate']
    sns.barplot(x='chatbot', y='error_rate', data=error_rate)
    plt.savefig('error_rate.png', bbox_inches='tight')
    plt.close()

    


