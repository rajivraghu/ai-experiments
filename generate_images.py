from PIL import Image, ImageDraw, ImageFont
import os
import glob

def get_font():
    fonts = glob.glob("/usr/share/fonts/**/*.ttf", recursive=True)
    if fonts:
        return ImageFont.truetype(fonts[0], 60)
    return ImageFont.load_default()

def create_image(filename, bg_color, text_color, text="Good Morning!"):
    img = Image.new('RGB', (800, 600), color=bg_color)
    d = ImageDraw.Draw(img)
    font = get_font()

    bbox = d.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = (800 - text_width) / 2
    y = (600 - text_height) / 2

    d.text((x, y), text, fill=text_color, font=font)
    img.save(filename)

create_image('good_morning_1.png', '#FFD700', '#000000', "Good Morning! Have a great day!")
create_image('good_morning_2.png', '#87CEEB', '#FFFFFF', "Good Morning! Rise and shine!")
create_image('good_morning_3.png', '#FF69B4', '#FFFFFF', "Good Morning! Smile!")
