import argparse
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
from tqdm import tqdm
from datasets import Dataset
import multiprocessing
import os

model = None

def init_worker(device):
    global model
    model = ChatterboxTTS.from_pretrained(device=device)

def process_item(item):
    text, output_path = item
    wav = model.generate(text)
    ta.save(output_path, wav, model.sr)

def main(args):
    dataset = Dataset.from_json(args.input_file)

    tasks = []
    for k in dataset:
        text = k['question']
        for t in range(args.n):
            uid = k['user_id']
            did = k['dialogue_id']
            output_path = f"{args.output_dir}/{uid}_{did}_{t}.wav"
            tasks.append((text, output_path))

    if args.skip_existing:
        tasks = [task for task in tasks if not os.path.exists(task[1])]

    # ---- Sharding logic ----
    total_tasks = len(tasks)
    shard_size = (total_tasks + args.num_shards - 1) // args.num_shards
    start = args.shard_id * shard_size
    end = min(start + shard_size, total_tasks)
    tasks_to_process = tasks[start:end]

    print(f"[INFO] Shard {args.shard_id}/{args.num_shards} "
          f"processing {len(tasks_to_process)} tasks "
          f"(range {start}:{end}) on device {args.device} "
          f"with {args.num_processes} processes")

    with multiprocessing.Pool(processes=args.num_processes,
                              initializer=init_worker,
                              initargs=(args.device,)) as pool:
        list(tqdm(pool.imap(process_item, tasks_to_process), total=len(tasks_to_process)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate audio from text using ChatterboxTTS with multiprocessing and sharding."
    )
    parser.add_argument("--input_file", type=str,
                        default="/home/tkdrnjs0621/work/longcontextva/dataset/base_data/recall_qa_test_unseen.jsonl",
                        help="Input JSONL file with questions.")
    parser.add_argument("--output_dir", type=str,
                        default="/home/tkdrnjs0621/work/longcontextva/dataset/audio/recall_question_wavs",
                        help="Directory to save generated WAV files.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run the model on (e.g., 'cuda', 'cpu').")
    parser.add_argument("--n", type=int, default=5,
                        help="Number of variations to generate for each question.")
    parser.add_argument("--num_processes", type=int, default=4,
                        help="Number of processes to use for multiprocessing.")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip regeneration if file exists.")
    parser.add_argument("--shard_id", type=int, default=0,
                        help="Shard index (0-based).")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Total number of shards.")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)
