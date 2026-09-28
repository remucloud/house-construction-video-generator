from flask import Flask, render_template, request
from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps
import numpy as np
import uuid
from pathlib import Path
import imageio.v3 as iio

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "static" / "generated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate_video():
    if "photo" not in request.files:
        return "No image uploaded", 400

    file = request.files["photo"]
    if file.filename == "":
        return "No file selected", 400

    try:
        reference = Image.open(file.stream).convert("RGB")
    except Exception:
        return "Invalid image file", 400

    reference = reference.resize((1200, 900))
    output_name = f"construction_{uuid.uuid4().hex}.mp4"
    output_path = OUTPUT_DIR / output_name

    frames = generate_frames(reference)
    iio.imwrite(output_path.as_posix(), frames, fps=4, quality=8)

    return render_template("index.html", video_url=f"/static/generated/{output_name}")


def generate_frames(reference):
    width, height = reference.size
    stages = [
        "Site prep",
        "Foundation",
        "Frame",
        "Roofing",
        "Walls",
        "Windows & doors",
        "Finishing",
        "Final house",
    ]

    frames = []
    for index, stage in enumerate(stages):
        progress = index / (len(stages) - 1)
        background = make_background(width, height)
        frame = build_stage_frame(background, reference, stage, progress)
        frames.append(np.array(frame))

    for _ in range(10):
        frames.append(np.array(reference))

    return frames


def make_background(width, height):
    image = Image.new("RGB", (width, height), (116, 182, 255))
    draw = ImageDraw.Draw(image)

    draw.rectangle((0, int(height * 0.72), width, height), fill=(150, 176, 120))

    # Tree silhouette
    draw.ellipse((width - 240, 20, width - 60, 230), fill=(38, 114, 60))
    draw.rectangle((width - 155, 160, width - 125, int(height * 0.75)), fill=(55, 76, 40))

    # small shrubs
    for x in [50, 160, 270, 410, 560, 700, 860, 1010]:
        draw.ellipse((x, int(height * 0.76), x + 45, int(height * 0.90)), fill=(42, 125, 56))

    return image


def estimate_house_bbox(img):
    gray = ImageOps.grayscale(img)
    arr = np.array(gray)
    mask = arr < 220
    ys, xs = np.where(mask)

    if len(xs) == 0:
        return (int(img.width * 0.15), int(img.height * 0.18), int(img.width * 0.82), int(img.height * 0.82))

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    pad_x = max(35, int(img.width * 0.08))
    pad_y = max(35, int(img.height * 0.10))

    return (
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(img.width, x1 + pad_x),
        min(img.height, y1 + pad_y),
    )


def build_stage_frame(background, reference, stage, progress):
    image = background.copy()
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = estimate_house_bbox(reference)
    left, top, right, bottom = x0, y0, x1, y1
    house_w = right - left
    house_h = bottom - top

    # base slab
    slab_y = int(image.height * 0.72)
    draw.rectangle((0, slab_y, image.width, image.height), fill=(122, 122, 118))

    if stage == "Site prep":
        draw.rectangle((left, bottom - 18, right, bottom), fill=(155, 155, 150), outline=(90, 90, 90), width=2)
        draw.line((left, bottom, right, bottom), fill=(90, 90, 90), width=4)

    elif stage == "Foundation":
        draw.rectangle((left, top + int(house_h * 0.68), right, bottom), fill=(145, 145, 142), outline=(85, 85, 85), width=2)
        draw.rectangle((left, top + int(house_h * 0.52), right, top + int(house_h * 0.68)), fill=(110, 108, 106))

    elif stage == "Frame":
        draw.rectangle((left, top + int(house_h * 0.18), right, bottom), fill=(190, 188, 182), outline=(80, 80, 80), width=2)
        for i in range(6):
            x = left + int((i + 1) * house_w / 7)
            draw.rectangle((x, top, x + 10, bottom), fill=(85, 85, 85))
        draw.polygon([(left - 20, top + int(house_h * 0.18)), (right + 20, top + int(house_h * 0.18)), (right, top), (left, top)], fill=(52, 52, 52))

    elif stage == "Roofing":
        draw.rectangle((left, top + int(house_h * 0.12), right, bottom), fill=(205, 210, 195), outline=(70, 70, 70), width=2)
        draw.polygon([(left - 15, top + int(house_h * 0.16)), (right + 15, top + int(house_h * 0.16)), (right, top), (left, top)], fill=(55, 55, 55))

    elif stage == "Walls":
        draw.rectangle((left, top + int(house_h * 0.08), right, bottom), fill=(218, 224, 200), outline=(70, 70, 70), width=2)
        draw.rectangle((left, top, right, top + int(house_h * 0.08)), fill=(60, 60, 60))
        # windows as dark blocks
        for wx in [left + 60, left + 250, left + 440, right - 250, right - 60]:
            draw.rectangle((wx, top + 80, wx + 100, bottom - 80), fill=(20, 20, 24))

    elif stage == "Windows & doors":
        draw.rectangle((left, top + int(house_h * 0.08), right, bottom), fill=(228, 232, 220), outline=(70, 70, 70), width=2)
        draw.rectangle((left + 250, bottom - 170, right - 250, bottom), fill=(120, 96, 75))
        for wx in [left + 45, left + 255, left + 465, right - 250, right - 60]:
            draw.rectangle((wx, top + 80, wx + 100, bottom - 80), fill=(20, 20, 24))
        draw.rectangle((left + 10, top + 70, left + 80, bottom - 70), fill=(18, 18, 18))

    elif stage == "Finishing":
        draw.rectangle((left, top + int(house_h * 0.08), right, bottom), fill=(230, 234, 220), outline=(70, 70, 70), width=2)
        # balcony and facade details
        draw.rectangle((left + 60, top + 120, right - 60, top + 180), fill=(160, 160, 160))
        for wx in [left + 50, left + 260, left + 470, right - 260, right - 60]:
            draw.rectangle((wx, top + 85, wx + 118, bottom - 90), fill=(18, 18, 22))
        draw.rectangle((left + 240, bottom - 155, right - 240, bottom), fill=(120, 90, 80))

    else:
        image = reference.copy()

    # label overlay
    label_box = (20, 20, 250, 58)
    draw.rounded_rectangle(label_box, radius=12, fill=(25, 25, 25, 160))
    try:
        font = ImageFont.load_default()
        draw.text((35, 28), stage, fill=(255, 255, 255), font=font)
    except Exception:
        pass

    # gradually bring in final reference photo
    if progress > 0.10:
        ref_layer = reference.copy().convert("RGBA")
        ref_layer = ImageEnhance.Brightness(ref_layer).enhance(0.6 + progress * 0.5)
        ref_layer = ImageEnhance.Contrast(ref_layer).enhance(0.8 + progress * 0.5)
        image = Image.blend(image.convert("RGBA"), ref_layer, 0.15 + progress * 0.25)
        image = image.convert("RGB")

    return image


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
