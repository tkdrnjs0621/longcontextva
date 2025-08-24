import argparse
import os
import soundfile as sf

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from datasets import Dataset
from tqdm import tqdm
import torch



def main():
    parser = argparse.ArgumentParser(description="Run inference with Qwen-VL-Audio model.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Omni-7B", help="Name of the pretrained model to use.")
    parser.add_argument("--data_path", type=str, default="/home/tkdrnjs0621/work/lcva/retrieval_results/retrieval_results_dialogue_asr.jsonl", help="Path to the input data JSONL file.")
    parser.add_argument("--audio_dir", type=str, default="/home/tkdrnjs0621/work/lcva/recall_wavs", help="Directory containing the audio files.")
    parser.add_argument("--output_path", type=str, default="/home/tkdrnjs0621/work/lcva/outputs/v1", help="Path to save the output audio and text files.")
    parser.add_argument("--prompt_type", type=str, default="v1", choices=["vanilla", "v1"], help="Type of prompt to use.")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_path, exist_ok=True)

    # Load the model on the available device(s)
    print(f"Loading model: {args.model_name}...")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(args.model_name, torch_dtype="auto", device_map="auto")
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_name)
    print("Model and processor loaded.")

    print(f"Loading data from {args.data_path}...")
    data = Dataset.from_json(args.data_path)
    
    print("Starting inference...")
    for row in tqdm(data):
        uid = row['user_id']
        did = row['dialogue_id']
        retrieved = row["retrieved_text"]
        audio_file = os.path.join(args.audio_dir, f"{uid}_{did}.wav")

        if not os.path.exists(audio_file):
            print(f"Warning: Audio file not found for {uid}_{did}.wav, skipping.")
            continue

        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_file},
                ],
            },
        ]

        # Preparation for inference
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        
        if args.prompt_type == "v1":
            text[0] += f"Considering the past utterance {retrieved}, My answer would be"
      
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        inputs = processor(text=text, audio=audios, return_tensors="pt", padding=True, use_audio_in_video=False)
        inputs = inputs.to(model.device).to(model.dtype)

        # Inference: Generation of the output text and audio

        with torch.no_grad():
            text_ids, audio = model.generate(**inputs)

        # Decode and save text output
        decoded_text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        text_output_file = os.path.join(args.output_path, f"{uid}_{did}.txt")
        with open(text_output_file, 'w', encoding='utf-8') as f:
            f.write(decoded_text)

        # Save audio output
        audio_output_file = os.path.join(args.output_path, f"{uid}_{did}.wav")
        sf.write(
            audio_output_file,
            audio.reshape(-1).detach().cpu().numpy(),
            samplerate=24000,
        )

        del inputs, text_ids, audio
        torch.cuda.empty_cache()
    print("Inference complete.")

if __name__ == "__main__":
    main()