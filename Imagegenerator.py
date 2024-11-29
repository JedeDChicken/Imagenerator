import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk # Helps render image from stable diffusion back to our app

# from authtoken import auth_token  # From Huggingface acc
# from huggingface_hub import HfApi
# api = HfApi()
# api.set_access_token('')

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, DiffusionPipeline

# App
app = tk.Tk()
app.geometry('532x632')
app.title('Imagenerator')
ctk.set_appearance_mode('dark')

prompt = ctk.CTkEntry(app, height=40, width=512, font=('Arial', 20), text_color='black', fg_color='white')  # Prompt
prompt.place(x=10, y=10)    

lmain = ctk.CTkLabel(app, height=512, width=512, text='')  # Frame, image placeholder
lmain.place(x=10, y=110)

# Stable Diffusion
# model_id = 'CompVis/stable-diffusion-v1-4'
model_id = 'stabilityai/stable-diffusion-2-1'
device = 'cuda'

# pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, revision='fp16', use_auth_token=auth_token)  # This revision allows loading into GPU w/ 4GB VRAM, dtype not dtypes
# pipe = pipe.to(device)  # Send pipe to GPU

# Load both base & refiner
# base = DiffusionPipeline.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0', torch_dtype=torch.float16, variant='fp16', use_safetensors=True)
# base.to(device)

# refiner = DiffusionPipeline.from_pretrained(
#     'stabilityai/stable-diffusion-xl-refiner-1.0',
#     text_encoder_2=base.text_encoder_2,
#     vae=base.vae,
#     torch_dtype=torch.float16,
#     use_safetensors=True,
#     variant='fp16',
# )
# refiner.to(device)

# Define how many steps and what % of steps to be run on each experts (80/20) here
# n_steps = 40
# high_noise_frac = 0.8

# pipe = DiffusionPipeline.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0', torch_dtype=torch.float16, use_safetensors=True, variant='fp16')
# pipe.to(device)

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, variant='fp16')
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

# prompt = 'a photo of an astronaut riding a horse on mars'
# image = pipe(prompt).images[0]
# image.save('astronaut_rides_horse.png')

# Generate Function
def generate():
    # pass
    with autocast("cuda"):  #Send to GPU
        # image = pipe(prompt.get(), guidance_scale=8.5)["sample"][0]   # Guiding scale defines how close we want stable diffusion to follow the prompt (higher is stricter)
        
        # image = base(
        #     prompt=prompt,
        #     num_inference_steps=n_steps,
        #     denoising_end=high_noise_frac,
        #     output_type="latent",
        # ).images
        # image = refiner(
        #     prompt=prompt,
        #     num_inference_steps=n_steps,
        #     denoising_start=high_noise_frac,
        #     image=image,
        # ).images[0]
        # image = pipe(prompt=prompt.get()).images[0]
        
        image = pipe(prompt.get(), guidance_scale=8.5).images[0]

    image.save('generated_image.png')
    img = ImageTk.PhotoImage(image.resize((512, 512))) # Get image, then we set the Frame
    lmain.configure(image=img)
    
trigger = ctk.CTkButton(app, height=40, width=120, font=('Arial', 20), text_color='white', fg_color='blue', command=generate)
trigger.configure(text='Generate')
x_trigger = 512/2 - 120/2
trigger.place(x=x_trigger, y=60)  # Button

app.mainloop()
# auto-py-to-exe

# Github- https://youtu.be/0e6LqN2LvRM
# Create new repository online
# Initialize repository on vscode, use actions...