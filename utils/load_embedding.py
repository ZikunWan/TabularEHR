import torch

_embedding_cache = None
_text_dim = None

def load_embedding_cache(cache_path: str):
    """
    Load pre-computed embedding cache for the given dataset.
    
    Args:
        cache_path (str): The correct absolute path to the dataset's embedding dictionary 
                          (e.g., /home/ma-user/.../embeddings/renji/text_embeddings.pt)
    
    Returns:
        tuple: (_embedding_cache, _text_dim)
    """
    global _embedding_cache, _text_dim
    
    if _embedding_cache is None:
        try:
            data = torch.load(cache_path, map_location='cpu', weights_only=False)
            _embedding_cache = data['embeddings']
            _text_dim = data['text_dim']
            print(f"Loaded {len(_embedding_cache)} embeddings (dim={_text_dim}) from {cache_path}")
        except Exception as e:
            print(f"Error loading embedding cache from {cache_path}: {e}")
            raise
            
    return _embedding_cache, _text_dim


def get_embedding(text: str) -> torch.Tensor:
    """
    Look up the embedding for a given text literal. 
    Returns the [PAD] token's zeroed embedding if the text isn't found.
    """
    global _embedding_cache, _text_dim
    
    if _embedding_cache is None:
         raise RuntimeError("Embedding cache not loaded. Please call `load_embedding_cache(path)` first.")
         
    return _embedding_cache.get(text, _embedding_cache.get('[PAD]', torch.zeros(_text_dim)))
