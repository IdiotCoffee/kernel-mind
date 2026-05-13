from datasets import Dataset


def build_ragas_dataset(samples):

    dataset_dict = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    for sample in samples:
        dataset_dict["question"].append(sample["question"])

        dataset_dict["answer"].append(sample["answer"])

        dataset_dict["contexts"].append(sample["contexts"])

        dataset_dict["ground_truth"].append(sample["ground_truth"])

    return Dataset.from_dict(dataset_dict)
