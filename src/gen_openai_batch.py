import json
import argparse
from datasets import load_dataset
from tqdm import tqdm

def format_dataset(dialogue_dataset):
    """Formats the raw dataset into a nested dictionary of dialogue strings."""
    formatted = {}
    for item in tqdm(dialogue_dataset['test'], desc="Formatting dialogues"):
        user_id = item['user_id']
        dialogue_id = item['dialogue_id']
        dialogue_type = item['type']
        if(dialogue_type != 'persona'):
            continue
        
        if user_id not in formatted:
            formatted[user_id] = {}
        if dialogue_id not in formatted[user_id]:
            formatted[user_id][dialogue_id] = []
        
        formatted[user_id][dialogue_id].append(item['user'])
        formatted[user_id][dialogue_id].append(item['assistant'])
        
    formatted_str = {}
    for uid, dialogs in formatted.items():
        formatted_str[uid] = {}
        for did, turns in dialogs.items():
            lines = []
            for i, text in enumerate(turns):
                prefix = 'User:' if i % 2 == 0 else 'Assistant:'
                lines.append(f'{prefix} {text}')
            formatted_str[uid][did] = "\n".join(lines)
            
    return formatted_str

def create_batch_file(formatted_dialogues, num_users=None, output_filename="batch_test.jsonl"):
    """Creates a JSONL file for the OpenAI Batch API."""
    
    system_prompt = """You will be given a conversation between the user and the assistant.
Your task is to create a questions and answers based on this conversation, speicifically, a **recall question** that asks about something said in the dialogue.
The output format must be:

```
Q: {recall question}  
A: {answer for recall question}
```

* The **recall question** should be phrased like: *"What did I/you say about …"* and must directly reference something from the original conversation.
* The question must be strongly tied to the given dialogue so that, even after many later conversations, asking the question would naturally make the assistant recall this specific dialogue.
* The question must be answer properly **only if** the given conversation is used to answer the question.
* Make sure the question sounds as natural as possible—something the user might genuinely forget and later ask the assistant about after some time.
* The question is asked by the user, and the answer is answered by assistant, so use the proper pronoun for question and answer.
""".strip()

    jobs = []
    
    user_ids_to_process = list(formatted_dialogues.keys())
    if num_users is not None and num_users > 0:
        user_ids_to_process = user_ids_to_process[:num_users]

    for uid in tqdm(user_ids_to_process, desc=f"Processing {len(user_ids_to_process)} users"):
        for did in formatted_dialogues[uid]:
            dialogue_text = formatted_dialogues[uid][did]
            
            job = {
                "custom_id": f"{uid}_{did}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4.1",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": dialogue_text}
                    ],
                    "temperature": 1,
                    "top_p": 1,
                }
            }
            jobs.append(job)

    with open(output_filename, 'w') as f:
        for job in tqdm(jobs, desc=f"Writing {len(jobs)} jobs to {output_filename}"):
            f.write(json.dumps(job) + '\n')
            
    return len(jobs)

def main():
    """Main function to run the data generation process."""
    parser = argparse.ArgumentParser(description="Generate a batch.jsonl file for OpenAI API from the HiCUPID dataset.")
    parser.add_argument("--num_users", type=int, default=None, help="Number of user_ids to process. Processes all users by default if not specified.")
    args = parser.parse_args()

    print("Loading dataset...")
    dialogue_dataset = load_dataset("12kimih/HiCUPID", "dialogue")
    
    print("Formatting dataset...")
    formatted_dialogues = format_dataset(dialogue_dataset)
    
    print("Creating batch file...")
    num_jobs_created = create_batch_file(formatted_dialogues, args.num_users)
    
    print(f"\nSuccessfully created batch.jsonl with {num_jobs_created} API requests.")

if __name__ == "__main__":
    main()