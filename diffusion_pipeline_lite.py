import argparse, torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

def load_pipeline(model="stabilityai/stable-diffusion-2-1"):
    pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16, safety_checker=None)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()
    return pipe

def generate(pipe, prompt, steps=30, guidance=7.5, seed=None):
    gen = torch.Generator("cuda").manual_seed(seed) if seed else None
    return pipe(prompt=prompt, num_inference_steps=steps, guidance_scale=guidance, generator=gen).images[0]

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--output", default="output.png")
    a = p.parse_args()
    pipe = load_pipeline()
    img = generate(pipe, a.prompt, a.steps)
    img.save(a.output)
    print(f"Saved: {a.output}")
