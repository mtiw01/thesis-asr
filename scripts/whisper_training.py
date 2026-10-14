import whisper
import torch

SIZE = "medium" # medium or turbo


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = whisper.load_model(SIZE, device=device)
    return model



def main():
    model = load_model()
    
    


if __name__=="__main__":
    main()



# # https://huggingface.co/blog/fine-tune-whisper#load-dataset

# from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor

# SIZE = "medium"

# feature_extractor = WhisperFeatureExtractor.from_pretrained(f"openai/whisper-{SIZE}")
# tokenizer = WhisperTokenizer.from_pretrained(f"openai/whisper-{SIZE}", language="English", task="transcribe")

# # test to check if tokenizer encodes/decodes correctly
# input_str = "hello everyone this is tag alshehri on behalf of my team i'm pleased to present our chi paper on qualitative secondary analysis for cross culture design"
# labels = tokenizer(input_str).input_ids
# decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
# decoded_str = tokenizer.decode(labels, skip_special_tokens=True)
# print(f"Input:                 {input_str}")
# print(f"Decoded w/ special:    {decoded_with_special}")
# print(f"Decoded w/out special: {decoded_str}")
# print(f"Are equal:             {input_str == decoded_str}")

# #processor = WhisperProcessor.from_pretrained(f"openai/whisper-{SIZE}", language="Hindi", task="transcribe")


