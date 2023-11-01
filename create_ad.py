from PIL import Image, ImageDraw, ImageFont
import textwrap
import torch
import cv2
from diffusers import StableDiffusionImg2ImgPipeline, EulerDiscreteScheduler
from utils import  adjust_font_size_for_text, load_img
# !pip install -q transformers diffusers accelerate torch==1.13.1
# !pip install -q "ipywidgets>=7,<8" ftfy



def init_stable_diffusion():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    model_id = "stabilityai/stable-diffusion-2"
    # Use the Euler scheduler here instead of default
    scheduler = EulerDiscreteScheduler.from_pretrained(
        model_id, subfolder="scheduler")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id, scheduler=scheduler, torch_dtype=torch.float16)
    pipe = pipe.to(device)

    return  pipe, device



def make_stable_diffusion(numpy_img,  hex_code, text_prompt="" ):
    pipe,device=init_stable_diffusion()

    img_rgb = cv2.cvtColor(numpy_img, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(img_rgb).convert("RGB").resize((768, 768))
    init_images = [pil_image]

    prompts = ["copy this image, " + text_prompt + "use " + hex_code]
    negative_prompts = ["not in different artstyle, not in different colors"]
    steps = 20
    scale = 9
    num_images_per_prompt = 1
    seed = torch.randint(0, 1000000, (1,)).item()
    generator = torch.Generator(device=device).manual_seed(seed)

    output = pipe(prompts, negative_prompt=negative_prompts, image=init_images, num_inference_steps=steps,
                  guidance_scale=scale, num_images_per_prompt=num_images_per_prompt, generator=generator)

    for idx, (image, prompt) in enumerate(zip(output.images, prompts * num_images_per_prompt)):
        image_name = str(idx) + ".png"
        print(image_name)
        image_path = "/home/enes/lab/case_study/" + image_name
        image.save(image_path, 'png')
    print("type(output.images): ", type(output.images))
    return output.images


def generate_ad_from_image_and_logo(template_img, logo, punchline, hex_code_punchline, button_text, hex_code_button):
    # Create a drawing context
    print("type(template_img): ", type(template_img))
    print("type(logo): ", type(logo))

    print("size template_img: ", template_img.size)
    print("size logo: ", logo.size)

    draw = ImageDraw.Draw(template_img)

    # Place the logo at the top center
    logo_width, logo_height = logo.size
    logo_position = ((template_img.width - logo_width) // 2, 10)
    template_img.paste(logo, logo_position, logo)

    font_path = "/home/enes/Lora-Italic-VariableFont_wght.ttf"
    font_size = 30
    font = ImageFont.truetype(font_path, font_size)

    # Draw a button at the bottom
    button_margin = 5
    max_button_width = 0.8 * template_img.width
    button_width = max_button_width
    button_height = 50
    button_position = ((template_img.width - button_width) // 2, template_img.height - button_height - button_margin)
    draw.rectangle([button_position, (button_position[0] + button_width, button_position[1] + button_height)], fill=hex_code_button)

    # Calculate the maximum width for the punchline
    max_punchline_width = 0.8 * template_img.width

    # Wrap the text using textwrap module
    wrapped_punchline = textwrap.wrap(punchline, width=20)

    # Calculate the total height of the wrapped punchline
    # total_text_height = sum([font.getsize(line)[1] for line in wrapped_punchline])
    total_text_height = sum(
        [draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1] for line in
         wrapped_punchline])

    # Calculate the starting Y position for the punchline
    text_y_position = button_position[1] - total_text_height - button_margin

    # Draw each line of the punchline
    for line in wrapped_punchline:
        text_bbox = draw.textbbox((0, 0), line, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        text_position = ((template_img.width - text_width) // 2, text_y_position)
        draw.text(text_position, line, font=font, fill=hex_code_punchline)
        text_y_position += text_height

    max_button_text_width = button_width - 2 * button_margin
    button_text_font = adjust_font_size_for_text(button_text, max_button_text_width, font_size, font_path, draw)

    # Place the button text in the middle of the button
    text_bbox = draw.textbbox((0, 0), button_text, font=button_text_font)
    button_text_width = text_bbox[2] - text_bbox[0]
    button_text_height = text_bbox[3] - text_bbox[1]
    button_text_position = (button_position[0] + (button_width - button_text_width) // 2, button_position[1] + (button_height - button_text_height) // 2)
    draw.text(button_text_position, button_text, font=button_text_font, fill="white")

    output_path = "/home/enes/lab/case_study/reklam.png"
    template_img.save(output_path, format='PNG')


    return template_img


def test_create_ad():

    hex_code_image = "#3A7BCD"
    hex_code_button = "#34ebab"
    hex_code_punchline = "#eb345f"
    punchline = "AI add banners lead to higher conversion rates"
    button_text = "Call to action text here!"

    logo_img_path = "/home/enes/lab/case_study/pepsi_logo.png"
    base_img_path = "base_image_sample.jpg"
    base_img = load_img(base_img_path, cv2.COLOR_BGR2RGB)
    base_img = cv2.resize(base_img, (256, 256))  # resize to 300x200

    logo = cv2.imread(logo_img_path, cv2.IMREAD_UNCHANGED)
    logo = cv2.resize(logo, (100, 100))  # resize to 300x200

    diffused_img = make_stable_diffusion(base_img, hex_code_image )
    diffused_img=diffused_img[0] # there is only 1 image so i am taking it


    new_dimensions = (350, 350)  # Specify new width and height
    diffused_img = diffused_img.resize(new_dimensions)

    logo = Image.fromarray(logo).convert('RGBA')
    new_dimensions = (64, 64)  # Specify new width and height
    logo = logo.resize(new_dimensions)

    generate_ad_from_image_and_logo(diffused_img, logo, punchline,  hex_code_punchline, button_text, hex_code_button)




if __name__ == "__main__":
   # test_diff()
   test_create_ad()

