import torch
import os
from sacrebleu import corpus_bleu
from tqdm import tqdm

from config import Config
from src.model.transformer import Transformer
from src.data.dataset import load_translation_data, create_dataloaders
from src.utils.tokenizer import load_tokenizers
from inference import TranslationInference

def evaluate_model(model_path, config, num_samples=1000):
    print("Loading test data...")
    train_data, val_data = load_translation_data(config)
    
    if len(val_data) > num_samples:
        test_samples = val_data.select(range(num_samples))
    else:
        test_samples = val_data
    
    print(f"Evaluating on {len(test_samples)} samples...")
    
    translator = TranslationInference(model_path, config)
    
    references = []
    predictions = []
    
    for i, sample in enumerate(tqdm(test_samples, desc="Translating")):
        src_text = sample['src']
        ref_text = sample['tgt']
        
        pred_text = translator.translate(src_text, max_length=100, beam_size=1)
        
        references.append([ref_text])
        predictions.append(pred_text)
        
        if i < 5:
            print(f"\nSample {i+1}:")
            print(f"Source: {src_text}")
            print(f"Reference: {ref_text}")
            print(f"Prediction: {pred_text}")
    
    bleu_score = corpus_bleu(predictions, references)
    print(f"\nBLEU Score: {bleu_score.score:.2f}")
    
    return bleu_score.score

def main():
    config = Config()
    model_path = os.path.join(config.model_path, 'best_model.pt')
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train the model first using train.py")
        return
    
    evaluate_model(model_path, config)

if __name__ == "__main__":
    main()