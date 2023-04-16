import pandas as pd
import itertools
import random
import numpy as np

"""
Generic dataset utils.
"""


def get_dataset(root: str, is_test=False):
    """
    Get dataset as a pandas dataframe.
    :param root: folder where the .csv files are located
    :param is_test: whether to get test or train dataset
    :return: pandas dataframe
    """
    name = "test.csv" if is_test else "train.csv"
    df = pd.read_csv(root + name)
    images_folder = "test_images/" if is_test else "train_images/"
    df['image'] = root + images_folder + df['image']
    return df


def add_target(df: pd.DataFrame):
    """
    Adds target column to the pandas dataframe.
    :param df: pandas dataframe for training data.
    :return: df but with added target column.
    """
    grouped = df.groupby('label_group')['posting_id'].apply(list)
    target = df['label_group'].map(grouped)
    new_df = df.copy()
    new_df['target'] = target
    return new_df


def get_contrastive_loss_dataset(df: pd.DataFrame, seed=42):
    random.seed(seed)
    label_groups = df['label_group'].unique().tolist()
    postings_per_group = df.groupby('label_group')['posting_id'].unique().to_dict()
    images_per_posting = df.groupby('posting_id')['image'].unique().to_dict()  # should be only one value per posting
    titles_per_posting = df.groupby('posting_id')['title'].unique().to_dict()  # should be only one value per posting
    new_data = []

    from tqdm import tqdm

    def get_negative_examples(cur_group_postings, excluded_index, num):
        sample_group_ids = list(range(len(label_groups)))
        sample_group_ids.remove(excluded_index)
        negative_group_ids = random.sample(sample_group_ids, num)
        negative_postings = [
            random.choice(postings_per_group[label_groups[neg_ind]].tolist())
            for neg_ind in negative_group_ids
        ]
        neg_examples = [(x_1, x_2, 0) for x_1, x_2 in itertools.product(cur_group_postings, negative_postings)]
        return neg_examples

    for i, label_group in tqdm(enumerate(label_groups)):
        group_postings = postings_per_group[label_group].tolist()
        positive_examples = [(x_1, x_2, 1) for x_1, x_2 in itertools.combinations(group_postings, 2) if x_1 != x_2]
        number_of_neg_examples = len(group_postings)
        negative_examples = get_negative_examples(group_postings, i, number_of_neg_examples)
        random.shuffle(negative_examples)
        negative_examples = negative_examples[:len(positive_examples)]  # let's keep ratio of positive to negatives ~ 1:1
        examples = positive_examples + negative_examples
        examples_with_titles_and_images = [
            (example[0],
             example[1],
             images_per_posting[example[0]].tolist()[0],
             images_per_posting[example[1]].tolist()[0],
             titles_per_posting[example[0]].tolist()[0],
             titles_per_posting[example[1]].tolist()[0],
             example[2]
             )
            for example
            in examples
        ]
        new_data.extend(examples_with_titles_and_images)
    columns = ['posting_id_1', 'posting_id_2', 'image_1', 'image_2', 'title_1', 'title_2', 'label']
    contrastive_loss_df = pd.DataFrame(new_data, columns=columns)
    return contrastive_loss_df
