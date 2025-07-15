import torch
import torch.nn.functional as F
import os
import argparse

from config import Config
from src.model.transformer import Transformer, create_padding_mask
from src.utils.tokenizer import load_tokenizers

class TranslationInference:
    def __init__(self, model_path, config):
        self.config = config
        self.device = config.device
        
        print("Loading tokenizers...")
        self.src_tokenizer, self.tgt_tokenizer = load_tokenizers(config)
        
        print("Loading model...")
        self.model = Transformer(
            src_vocab_size=self.src_tokenizer.get_piece_size(),
            tgt_vocab_size=self.tgt_tokenizer.get_piece_size(),
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout
        ).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print("Model loaded successfully!")
    
    def translate(self, src_text, max_length=100, beam_size=1):
        with torch.no_grad():
            if beam_size == 1:
                return self._greedy_decode(src_text, max_length)
            else:
                return self._beam_search(src_text, max_length, beam_size)
    
    def _greedy_decode(self, src_text, max_length):
        src_tokens = [self.src_tokenizer.bos_id()] + \
                    self.src_tokenizer.encode(src_text, out_type=int) + \
                    [self.src_tokenizer.eos_id()]
        
        src_tensor = torch.tensor([src_tokens], dtype=torch.long).to(self.device)
        src_mask = create_padding_mask(src_tensor, pad_idx=0).to(self.device)
        
        encoder_output = self.model.encode(src_tensor, src_mask)
        
        tgt_tokens = [self.tgt_tokenizer.bos_id()]
        
        for _ in range(max_length):
            tgt_tensor = torch.tensor([tgt_tokens], dtype=torch.long).to(self.device)
            tgt_mask = create_padding_mask(tgt_tensor, pad_idx=0).to(self.device)
            
            decoder_output = self.model.decode(tgt_tensor, encoder_output, src_mask, tgt_mask)
            output = self.model.output_projection(decoder_output)
            
            next_token = output[0, -1, :].argmax().item()
            tgt_tokens.append(next_token)
            
            if next_token == self.tgt_tokenizer.eos_id():
                break
        
        if tgt_tokens[-1] != self.tgt_tokenizer.eos_id():
            tgt_tokens.append(self.tgt_tokenizer.eos_id())
        
        translated_text = self.tgt_tokenizer.decode(tgt_tokens[1:-1])
        return translated_text
    
    def _beam_search(self, src_text, max_length, beam_size):
        src_tokens = [self.src_tokenizer.bos_id()] + \
                    self.src_tokenizer.encode(src_text, out_type=int) + \
                    [self.src_tokenizer.eos_id()]
        
        src_tensor = torch.tensor([src_tokens], dtype=torch.long).to(self.device)
        src_mask = create_padding_mask(src_tensor, pad_idx=0).to(self.device)
        
        encoder_output = self.model.encode(src_tensor, src_mask)
        
        beams = [(torch.tensor([[self.tgt_tokenizer.bos_id()]], dtype=torch.long).to(self.device), 0.0)]
        
        for _ in range(max_length):
            candidates = []
            
            for beam_tokens, beam_score in beams:
                if beam_tokens[0, -1].item() == self.tgt_tokenizer.eos_id():
                    candidates.append((beam_tokens, beam_score))
                    continue
                
                tgt_mask = create_padding_mask(beam_tokens, pad_idx=0).to(self.device)
                decoder_output = self.model.decode(beam_tokens, encoder_output, src_mask, tgt_mask)
                output = self.model.output_projection(decoder_output)
                
                log_probs = F.log_softmax(output[0, -1, :], dim=-1)
                top_k_probs, top_k_indices = torch.topk(log_probs, beam_size)
                
                for i in range(beam_size):
                    next_token = top_k_indices[i].item()
                    next_score = beam_score + top_k_probs[i].item()
                    
                    new_beam_tokens = torch.cat([
                        beam_tokens,
                        torch.tensor([[next_token]], dtype=torch.long).to(self.device)
                    ], dim=1)
                    
                    candidates.append((new_beam_tokens, next_score))
            
            beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]
            
            if all(beam[0][0, -1].item() == self.tgt_tokenizer.eos_id() for beam in beams):
                break
        
        best_beam = beams[0][0][0].tolist()
        if best_beam[0] == self.tgt_tokenizer.bos_id():
            best_beam = best_beam[1:]
        if best_beam[-1] == self.tgt_tokenizer.eos_id():
            best_beam = best_beam[:-1]
        
        translated_text = self.tgt_tokenizer.decode(best_beam)
        return translated_text

def main():
    parser = argparse.ArgumentParser(description='Chinese to English Translation')
    parser.add_argument('--model_path', type=str, default='./model/best_model.pt',
                       help='Path to the trained model')
    parser.add_argument('--text', type=str, required=True,
                       help='Chinese text to translate')
    parser.add_argument('--beam_size', type=int, default=1,
                       help='Beam size for beam search (1 for greedy)')
    parser.add_argument('--max_length', type=int, default=100,
                       help='Maximum length of generated translation')
    
    args = parser.parse_args()
    
    config = Config()
    
    if not os.path.exists(args.model_path):
        print(f"Model not found at {args.model_path}")
        print("Please train the model first using train.py")
        return
    
    translator = TranslationInference(args.model_path, config)
    
    print(f"Source (Chinese): {args.text}")
    translation = translator.translate(args.text, args.max_length, args.beam_size)
    print(f"Translation (English): {translation}")

if __name__ == "__main__":
    main()