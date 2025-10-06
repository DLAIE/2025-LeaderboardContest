# Submission Protocols 



Your code must include a section that implements/subclasses the `SubmissionInterface` with the following endpoints: 

```python
class SubmissionInterface:
    """All teams must implement this for automated evaluation.
    When you subclass/implement these methods, replace the NotImplementedError."""
    
    def __init__(self):
        self.vae, self.flow_model = None, None
    
    def load_vae(self, weights_path: str):
        """this completely specifies the vae model including configuration parameters,
           downloads/mounts the weights from Google Drive, automatically loads weights"""
        # self.vae = ...
        raise NotImplementedError
    
    def load_flow_model(self, weights_path: str):
        """this completely specifies the flow model including configuration parameters,
           downloads/mounts the weights from Google Drive, automatically loads weights"""
        # self.flow_model = ...
        raise NotImplementedError
    
    def generate_samples(self, n_samples: int) -> torch.Tensor:
        # returns (n_samples, 28, 28), normalized on [0,1]
        raise NotImplementedError
    
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.vae.encoder(images)
    
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.vae.decoder(latents)
```



For downloading weights, you can use this utlity function that takes a google *sharing* url, and works on either Colab or a laptop

```python
def download_weights(gdrive_url: str, local_path: str) -> str:
    """Download from Google Drive. Returns local_path."""
    import os
    if os.path.exists(local_path): return local_path
    try: import gdown
    except: 
      import subprocess; subprocess.check_call(['pip', 'install', '-q', 'gdown']); import gdown
    gdown.download(gdrive_url, local_path, quiet=False, fuzzy=True)
    return local_path
```

My evaluation code will run something like this (WIP): 
```python
import importlib.util
spec = importlib.util.spec_from_file_location("submission", "team_name_submission.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

submission = module.SubmissionInterface()
submission.load_vae('vae_weights.safetensors')
submission.load_flow_model('flow_weights.safetensors')
samples = submission.generate_samples(100)
# etc etc etc 
```
