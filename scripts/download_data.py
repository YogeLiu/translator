from datasets import load_dataset
from config import Config

def download_dataset():
    config = Config()
    print(f"Downloading dataset: {config.dataset_name}")
    
    try:
        dataset = load_dataset(config.dataset_name)
        print(f"Dataset downloaded successfully!")
        print(f"Train samples: {len(dataset['train'])}")
        if 'validation' in dataset:
            print(f"Validation samples: {len(dataset['validation'])}")
        
        print("\nSample data:")
        sample = dataset['train'][0]
        print(f"Source (Chinese): {sample['src']}")
        print(f"Target (English): {sample['tgt']}")
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")

if __name__ == "__main__":
    download_dataset()