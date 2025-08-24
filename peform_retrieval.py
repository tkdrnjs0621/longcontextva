import argparse
import json
from collections import defaultdict

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from tqdm import tqdm

# Load E5-large (English). Use v2 if you prefer: "intfloat/e5-large-v2"
model_name = "intfloat/e5-large"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(model_name, device=device)

def embed_passages(passages, batch_size=64):
    # E5 expects 'passage:' prefix for documents
    prefixed = [f"passage: {p}" for p in passages]
    emb = model.encode(prefixed, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True)
    return emb  # shape: (N, d), L2-normalized

def embed_query(query):
    # E5 expects 'query:' prefix for queries
    q_emb = model.encode([f"query: {query}"], convert_to_numpy=True, normalize_embeddings=True)[0]
    return q_emb  # shape: (d,), L2-normalized

def top1_retrieve(query, corpus):
    """
    Returns (best_text, best_index, similarity_score)
    """
    if not corpus:
        raise ValueError("Corpus is empty.")
    doc_emb = embed_passages(corpus)
    q_emb = embed_query(query)

    # Cosine similarity == dot product since embeddings are normalized
    sims = doc_emb @ q_emb  # shape: (N,)
    best_idx = int(np.argmax(sims))
    return corpus[best_idx], best_idx, float(sims[best_idx])

def main():
    parser = argparse.ArgumentParser(description="Perform retrieval on the HiCUPID dataset.")
    parser.add_argument("--retrieval_unit", type=str, default='user', choices=["dialogue", "user"],
                        help="The unit for retrieval, either 'dialogue' or 'user'.")
    parser.add_argument("--retrieval_type", type=str, default='gt', choices=["gt","asr"],
                        help="The unit for retrieval, either 'dialogue' or 'user'.")
    parser.add_argument("--recall_dataset_path", type=str, default="output_wer.jsonl",
                        help="Path to the recall dataset file.")
    parser.add_argument("--output_path", type=str, default="retrieval_results_user.jsonl",
                        help="Path to save the retrieval results.")
    args = parser.parse_args()

    # Load the HiCUPID dataset
    print("Loading HiCUPID dataset...")
    hicupid_dataset = load_dataset("12kimih/HiCUPID", "dialogue")['train']

    # Create corpora based on the retrieval unit
    print(f"Creating corpus for retrieval unit: {args.retrieval_unit}...")
    if args.retrieval_unit == 'dialogue':
        corpus_map = defaultdict(list)
        for item in hicupid_dataset:
            corpus_map[item['dialogue_id']].append(item['user'])
            corpus_map[item['dialogue_id']].append(item['assistant'])
    elif args.retrieval_unit == 'user':
        corpus_map = defaultdict(list)
        for item in hicupid_dataset:
            corpus_map[item['user_id']].append(item['user'])
            corpus_map[item['user_id']].append(item['assistant'])
    else:
        raise ValueError(f"Invalid retrieval unit: {args.retrieval_unit}")

    # Load the recall dataset
    print(f"Loading recall dataset from {args.recall_dataset_path}...")
    recall_questions = []
    with open(args.recall_dataset_path, 'r') as f:
        for line in f:
            recall_questions.append(json.loads(line))

    # Perform retrieval and save results
    print(f"Performing retrieval and saving results to {args.output_path}...")
    with open(args.output_path, 'w') as f_out:
        for item in tqdm(recall_questions):
            query = item['question' if args.retrieval_type == 'gt' else 'asr_result']
            if args.retrieval_unit == 'dialogue':
                key = item['dialogue_id']
            else: # user
                key = item['user_id']
            
            corpus = corpus_map.get(int(key))
            if not corpus:
                print(f"Warning: No corpus found for {args.retrieval_unit} ID {key}. Skipping.")
                continue

            try:
                best_text, best_idx, score = top1_retrieve(query, corpus)
                result = {
                    "retrieval_unit": args.retrieval_unit,
                    "retrieval_type": args.retrieval_type,
                    "user_id": item['user_id'], 
                    "dialogue_id": item['dialogue_id'], 
                    "question": item['question'],
                    "question_asr": item['asr_result'],
                    "retrieved_text": best_text,
                    "gt_answer": item.get("answer")
                }
                f_out.write(json.dumps(result) + '\n')
            except ValueError as e:
                print(f"Error processing query '{query}': {e}")

    print("Retrieval complete.")

if __name__ == "__main__":
    main()