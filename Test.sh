import torch
from open_clip import create_model_and_transforms
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import openslide
from transformers import CLIPTokenizer

model_name = 'ViT-B-16'
checkpoint_path = './pathgen-clip-l.pt'
model, _, preprocess = create_model_and_transforms(model_name, pretrained=None)
state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
model.load_state_dict(state_dict)

# image = preprocess(Image.open("example.png")).unsqueeze(0)
# text = tokenizer(["An H&E image of tumor patch", "An H&E image of normal patch"])
slide = openslide.OpenSlide("./TCGA-AA-3844-01Z-00-DX1.bf88ce1f-0601-40c8-813e-4e3df51bd2f0.svs")
thumbnail = slide.get_thumbnail((512, 512)) 
image = preprocess(thumbnail).unsqueeze(0)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
#text = tokenizer(["An H&E image of tumor patch", "An H&E image of normal patch"], return_tensors="pt")
text = tokenizer(
    ["An H&E image of tumor patch", "An H&E image of normal patch"],
    padding="max_length",
    truncation=True,
    max_length=77,
    return_tensors="pt"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image = image.to(device)
text_input_ids = text["input_ids"].to(device)
model = model.to(device)

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text_input_ids)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)
#print("Image features:", image_features)
