from flask import Flask, request, render_template, send_file
from diffusers import DiffusionPipeline
import torch
from io import BytesIO

app = Flask(__name__)

# Load the fine-tuned model
model_path = "path_to_your_finetuned_model"
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


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_image():
    try:
        # Get the prompt from the user
        prompt = request.form.get('prompt')
        
        if not prompt:
            return "Prompt is required", 400
        
        # Generate the image
        image = pipe(prompt).images[0]

        # Save to a BytesIO object
        img_io = BytesIO()
        image.save(img_io, format='PNG')
        img_io.seek(0)

        # Return the image as a file
        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        return f"An error occurred: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)
