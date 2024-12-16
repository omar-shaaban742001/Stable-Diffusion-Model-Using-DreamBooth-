from diffusers import DiffusionPipeline
import matplotlib.pyplot as plt 

path = 'D:/model'


# Load the model with a custom cache directory
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", 
    cache_dir= path
)

# Load the LoRA weights with the same custom cache directory
pipe.load_lora_weights(
    "AdamLucek/sdxl-base-1.0-greenchair-dreambooth-lora", 
    cache_dir=path
)

# Use the pipeline as usual
prompt = "A photo of sks chair in an apartment"
result = pipe(prompt).images[0]
result.show()
