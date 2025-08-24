import argparse
import json
from datasets import Dataset

def main():
    parser = argparse.ArgumentParser(description='Process batch results.')
    parser.add_argument('--input_file', type=str, default='/home/tkdrnjs0621/work/lcva/dataset/batch_output/batch_recall_test_output.jsonl', help='The path to the input JSONL file.')
    parser.add_argument('--output_file', type=str, default='/home/tkdrnjs0621/work/lcva/dataset/base_data/recall_qa.jsonl', help='The path to the output JSONL file.')
    args = parser.parse_args()

    dataset = Dataset.from_json(args.input_file)
    
    ls = []
    for k in dataset:
        a = k['response']['body']['choices'][0]['message']['content']
        aa = a.split('Q:')[-1]
        q = aa.split('A:')[0].strip()
        a = aa.split('A:')[-1].strip()
        
        uid = k['custom_id'].split('_')[0]
        did = k['custom_id'].split('_')[-1]
        ls.append({'user_id':uid, 'dialogue_id':did, 'question':q, 'answer':a})

    Dataset.from_list(ls).to_json(args.output_file)

if __name__ == '__main__':
    main()
