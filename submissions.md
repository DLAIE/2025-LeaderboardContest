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


Here's a more complete example of a `SubmissionInterface`: 
```python
from safetensors.torch import load_file
import gdown

class SubmissionInterface:
    def __init__(self):
        self.device = 'cpu' 
        self.latent_dim = 3
        self.load_vae()
        self.load_flow_model()
    
    def load_vae(self):
        """this completely specifies the vae model including configuration parameters,
           downloads/mounts the weights from Google Drive, automatically loads weights"""
        safetensors_link = "https://drive.google.com/file/d/xxxx1/view?usp=sharing"
        output = 'downloaded_vae.safetensors'
        gdown.download(safetensors_link, output, quiet=False, fuzzy=True)
        self.vae = SimpleVAEModel(latent_dim=self.latent_dim)
        self.vae.load_state_dict(load_file(output))
        
    def load_flow_model(self):
        """this completely specifies the flow model including configuration parameters,
           downloads/mounts the weights from Google Drive, automatically loads weights"""
        safetensors_link = "https://drive.google.com/file/d/xxxx2/view?usp=sharing"
        output = 'downloaded_flow.safetensors'
        gdown.download(safetensors_link, output, quiet=False, fuzzy=True)
        self.flow_model = SimpleFlowModel(latent_dim=self.latent_dim)
        self.flow_model.load_state_dict(load_file(output))
        self.flow_model.to(self.device)
    
    def generate_samples(self, n_samples:int, n_steps:int) -> torch.Tensor:
        z0 = torch.randn([n_samples, self.latent_dim]).to(self.device)
        z1 = integrate_path(self.flow_model, z0, n_steps=n_steps)
        gen_xhat = F.sigmoid(vae.decoder(z1).view(-1, 28, 28))
        return gen_xhat

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        # TODO: we should somehow gracefully handle flattening for linear/conv layers. 
        return self.vae.encoder(images)
    
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.vae.decoder(latents)

    def to(self, device):
        self.device = device 
        self.vae.to(self.device)
        self.flow_model.to(self.device)
        return self


mysub = SubmissionInterface().to('cuda')
xhat_gen = mysub.generate_samples(n_samples=10, n_steps=100)
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
