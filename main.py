import io
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
import open_clip
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="Image Embedding API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms(
    model_name="ViT-B-32", pretrained="laion2b_s34b_b79k"
)
model = model.to(device)
model.eval()


def embed_image(image: Image.Image):
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image_tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy()[0].tolist()


# ===== Endpoint =====
@app.post("/image-embedding")
async def image_embedding(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    vector = embed_image(image)

    return {"dimension": len(vector), "embedding": vector}


@app.get("/")
def health():
    return {"status": "ok"}
