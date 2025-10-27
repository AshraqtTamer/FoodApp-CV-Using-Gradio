# 🍱 Food Classification using Transfer Learning (Sushi, Pizza, Steak)

This project applies **transfer learning** using **EfficientNetB0** and **Vision Transformer (ViT)** architectures to classify food images into three categories: **Sushi**, **Pizza**, and **Steak**.  
It demonstrates how to combine **custom datasets**, **fine-tuning**, and **modern transformer-based vision models** to achieve accurate and efficient food recognition.

---

## 🚀 Project Overview

Food recognition is an essential component in AI-powered food apps and restaurant recommendation systems.  
In this notebook, we:
- Load and preprocess a **custom image dataset**
- Use **transfer learning** with pre-trained models:
  - `EfficientNetB0`
  - `Vision Transformer (ViT)` from Hugging Face
- Fine-tune both models for a 3-class classification problem
- Compare performance and visualize results

---

## 🧠 Models Used

### 1. EfficientNetB0  
- Lightweight CNN architecture known for efficiency and strong performance.  
- Fine-tuned on custom dataset.

### 2. Vision Transformer (ViT)  
- Transformer-based architecture for image classification.  
- Loaded via Hugging Face:

```python
from transformers import ViTFeatureExtractor, ViTForImageClassification
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
```

---

## 📂 Dataset

The dataset contains three food categories:
- 🍣 **Sushi**  
- 🍕 **Pizza**  
- 🥩 **Steak**

Each category includes a set of labeled images stored in separate folders.

> ⚠️ For privacy and size constraints, the dataset and full notebook outputs are **not included in this repository**.  
> You can access them via Google Drive below.

---

## 🔗 Access the Full Notebook

Because of file size limits on GitHub, the complete notebook (with full outputs, plots, and model weights) is hosted on Google Drive:

👉 [**View Full Notebook on Google Drive**](https://drive.google.com/your-link-here)

*(Replace with your actual Drive share link — set access to “Anyone with the link can view.”)*

---

## 🛠️ Installation & Requirements

To run this notebook locally or in Google Colab:

```bash
# Clone this repository
git clone https://github.com/yourusername/FoodApp_with_transfer_learning.git
cd FoodApp_with_transfer_learning

# Install dependencies
pip install torch torchvision transformers timm numpy pandas matplotlib scikit-learn gradio
```

---

## 🧾 Usage

Open the notebook in Google Colab or Jupyter:

```bash
jupyter notebook FoodApp_with_transfere_learning_and_custom_Data_MyNootebook.ipynb
```

Steps inside the notebook:
1. Load and preprocess the dataset  
2. Apply transfer learning with EfficientNetB0 and ViT  
3. Fine-tune both models  
4. Evaluate accuracy, loss, and visualizations  

---

## 📊 Results Summary

| Model | Accuracy | Loss | Notes |
|:------|:----------:|:------:|:------|
| **EfficientNetB0** | ~95% | Low | Fast and lightweight |
| **Vision Transformer (ViT)** | **~99%** | Very Low | Achieved excellent performance after fine-tuning on Hugging Face |

*(Values are approximate and may vary based on dataset and epochs.)*

---

## 📈 Visualizations

- Training and validation loss curves  
- Confusion matrix  
- Sample predictions with true labels  

---

## 🚀 Deployment on Hugging Face Spaces with Gradio

You can deploy this model as an **interactive web app** using [**Gradio**](https://gradio.app/) and [**Hugging Face Spaces**](https://huggingface.co/spaces).

### 🧩 Steps

1. Create a new **Space** on Hugging Face (select type: **Gradio**).  
2. Upload your trained model files and a new script named `app.py`.  

Example `app.py`:

```python
import gradio as gr
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import torch

model = ViTForImageClassification.from_pretrained("your-huggingface-model")
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

labels = ["Sushi", "Pizza", "Steak"]

def predict(image):
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    return {labels[i]: float(probs[0][i]) for i in range(len(labels))}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="🍱 Food Classifier (ViT)",
    description="Classify food images as Sushi, Pizza, or Steak."
)

demo.launch()
```

3. Add a `requirements.txt` file with:
```text
torch
transformers
gradio
Pillow
```

4. Commit and push to your Hugging Face Space.  
Your app will automatically build and launch online.

---

## 🧩 Future Work
- Expand dataset with more food categories  
- Deploy via Streamlit or FastAPI as alternatives  
- Experiment with hybrid CNN–Transformer architectures  

---

## 👩‍💻 Author
**Ashraqt Tamer**  
AI & Deep Learning Enthusiast  

---

## 📜 License
This project is licensed under the **MIT License** — you’re free to use, modify, and distribute it with attribution.
