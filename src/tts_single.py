import argparse
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
from tqdm import tqdm
from datasets import Dataset

def main(args):
    model = ChatterboxTTS.from_pretrained(device=args.device)
    
    dataset = Dataset.from_json(args.input_file)
    for k in tqdm(dataset):
        text = k['question']
        
        for t in range(args.n):
            wav = model.generate(text)
            uid = k['user_id']
            did = k['dialogue_id']
            ta.save(f"{args.output_dir}/{uid}_{did}_{t}.wav", wav, model.sr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate audio from text using ChatterboxTTS.")
    parser.add_argument("--input_file", type=str, default="/home/tkdrnjs0621/work/lcva/dataset/base_data/recall_qa.jsonl", help="Input JSONL file with questions.")
    parser.add_argument("--output_dir", type=str, default="/home/tkdrnjs0621/work/lcva/dataset/recall_question_wavs", help="Directory to save generated WAV files.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (e.g., 'cuda', 'cpu').")
    parser.add_argument("--n", type=int, default=5, help="Device to run the model on (e.g., 'cuda', 'cpu').")
    
    args = parser.parse_args()
    main(args)
