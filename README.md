# 🖼️ Image Captioning with Hugging Face Transformers

This project demonstrates how to generate captions for images using a pretrained model from Hugging Face 🤗 — specifically, `nlpconnect/vit-gpt2-image-captioning`.

---

## 🧠 Model Used

- **Model**: [`nlpconnect/vit-gpt2-image-captioning`](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning)
- **Backbone**: Vision Transformer (ViT) + GPT-2

---

## 📷 Example Image & Caption

| Image | Generated Caption |
|-------|-------------------|
| ![Parrots](images/parrots.png) | "two parrots are sitting on a branch" |
| ![Bike](images/bike.jpg) | "a man riding a motorcycle on the street" |

---

## 📦 Installation

Install required Python libraries:

```bash
pip install transformers torch torchvision pillow matplotlib
