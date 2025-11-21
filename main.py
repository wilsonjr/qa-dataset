import argparse
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
    return 1-dataset.to_pandas().groupby("chatbot")["is_correct"].mean()

def extract_random_questions(dataset: Dataset, sample_rate: float):
    """
    Extracts random questions from the dataset according to the sample rate
    """
    if sample_rate > 1 or sample_rate < 0:
        raise ValueError(f"sample_rate must be between 0 and 1")

    buckets = defaultdict(list)
    categories = set(dataset['category'])
    for category in categories:
        subset = dataset.filter(lambda x: x['category'] == category)
        buckets[category].extend(subset.select(random.sample(range(len(subset)), int(sample_rate * len(subset)))))

    return buckets

def plot_error_rate(dataset: Dataset, save_path: str|None = None):
    """
    Plots the error rate by chatbot
    If save_path is provided, the plot is saved to the path
    Otherwise, the plot is displayed
    """
    error_rate = get_error_rate(dataset).reset_index()
    error_rate.columns = ['chatbot', 'error_rate']
    error_rate['error_rate'] = error_rate['error_rate']
    sns.barplot(x='chatbot', y='error_rate', data=error_rate, hue='chatbot')
    plt.title('Error Rate by Chatbot')
    plt.xlabel('Chatbot')
    plt.ylabel('Error Rate')
    plt.savefig(save_path or "error_rate.png", bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot_error_rate", action="store_true")
    parser.add_argument("--get_random_questions", action="store_true", default=False)
    parser.add_argument("--sample_rate", type=float, default=0.1)
    args = parser.parse_args()

    dataset = load_dataset("daloopa/financial-retrieval", split="train")
    
    if args.plot_error_rate:
        plot_error_rate(dataset)
    if args.get_random_questions:
        questions = extract_random_questions(dataset, args.sample_rate)
        print(questions)

if __name__ == "__main__":
    main()
    


