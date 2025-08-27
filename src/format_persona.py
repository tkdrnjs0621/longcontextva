import argparse
from datasets import Dataset, load_dataset

def main(split, save_path):
    dataset = load_dataset("12kimih/HiCUPID", "qa")[split]
    ls = []
    for row in dataset:
        if(row['type']!='persona'):
            continue
        ls.append(
            {
                'user_id': row['user_id'],
                'dialogue_id': row['dialogue_id'],
                'question' : row['question'],
                'answer': row['personalized_answer']
            })

    Dataset.from_list(ls).to_json(save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test_2")
    parser.add_argument("--save_path", type=str, default="base_data/persona_qa_test_unseen.jsonl")
    args = parser.parse_args()
    main(args.split, args.save_path)