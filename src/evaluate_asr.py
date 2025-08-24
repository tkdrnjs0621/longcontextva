
import argparse
import os
import whisper
import shutil
from collections import defaultdict
from datasets import Dataset
from evaluate import load as load_metric
import re
from tqdm import tqdm
import torch

def normalize_text(text):
    """Removes punctuation and converts to lowercase."""
    return re.sub(r'[^\w\s]', '', text).lower()

def main(args):
    """
    Main function to run the ASR evaluation and selection process.
    """
    # 1. Load models and metrics
    device = "cuda" if torch.cuda.is_available() else "cpu"
    whisper_model = whisper.load_model(args.model_name, device=device)
    wer_metric = load_metric("wer")

    # 2. Load reference dataset
    try:
        reference_dataset = Dataset.from_json(args.reference_dataset)
        reference_lookup = {
            f"{item['user_id']}_{item['dialogue_id']}": item['question']
            for item in reference_dataset
        }
    except FileNotFoundError:
        print(f"Error: Reference dataset not found at {args.reference_dataset}")
        return

    # 3. Group audio files by uid and did
    audio_files = defaultdict(list)
    for filename in os.listdir(args.input_dir):
        if filename.lower().endswith('.wav'):
            try:
                uid, did, _ = filename.rsplit('_', 2)
                group_key = f"{uid}_{did}"
                audio_files[group_key].append(os.path.join(args.input_dir, filename))
            except ValueError:
                print(f"Skipping file with unexpected format: {filename}")
                continue
    
    # 4. Create output directories
    os.makedirs(args.output_audio_dir, exist_ok=True)

    # 5. Process each group of audio files
    results = []
    for group_key, file_paths in tqdm(audio_files.items(), desc="Processing audio groups"):
        ground_truth = reference_lookup.get(group_key)
        if not ground_truth:
            print(f"Warning: No reference found for group {group_key}. Skipping.")
            continue

        best_wer = float('inf')
        best_audio_path = None
        best_transcription = ""

        for audio_path in file_paths:
            # Transcribe audio
            transcription_result = whisper_model.transcribe(audio_path, fp16=torch.cuda.is_available())
            transcribed_text = transcription_result['text']

            # Calculate WER
            normalized_gt = normalize_text(ground_truth)
            normalized_pred = normalize_text(transcribed_text)
            wer = wer_metric.compute(predictions=[normalized_pred], references=[normalized_gt])

            if wer < best_wer:
                best_wer = wer
                best_audio_path = audio_path
                best_transcription = transcribed_text

        # Save the best audio and the result
        if best_audio_path:
            uid, did = group_key.split('_')
            output_filename = f"{uid}_{did}.wav"
            output_path = os.path.join(args.output_audio_dir, output_filename)
            shutil.copy(best_audio_path, output_path)
            
            results.append({
                "user_id": uid,
                "dialogue_id": did,
                "best_audio_path": best_audio_path,
                "wer": best_wer,
                "transcription": best_transcription,
                "ground_truth": ground_truth,
            })

    # 6. Save results to a JSONL file
    Dataset.from_list(results).to_json(args.output_result_file, orient="records", lines=True)
    print(f"Processing complete. Best audio files saved to {args.output_audio_dir}")
    print(f"Evaluation results saved to {args.output_result_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ASR performance, select the best audio, and save results.")
    
    parser.add_argument("--input_dir", type=str, default="/home/tkdrnjs0621/work/lcva/dataset/recall_question_wavs", help="Directory with input WAV files (format: uid_did_num.wav).")
    parser.add_argument("--output_audio_dir", type=str, default="/home/tkdrnjs0621/work/lcva/outputs/best_recall_wavs", help="Directory to save the best performing WAV files.")
    parser.add_argument("--output_result_file", type=str, default="/home/tkdrnjs0621/work/lcva/output_wer.jsonl", help="Path to save the final WER results in JSONL format.")
    parser.add_argument("--reference_dataset", type=str, default="/home/tkdrnjs0621/work/lcva/dataset/base_data/recall_qa.jsonl", help="Path to the reference dataset with ground truth transcriptions.")
    parser.add_argument("--model_name", type=str, default="turbo", help="Name of the Whisper model to use (e.g., tiny, base, small).")

    args = parser.parse_args()
    main(args)
