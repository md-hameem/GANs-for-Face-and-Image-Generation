# 🎭 Anime Face Generator using DCGAN (PyTorch + Streamlit)

A professional AI web app that generates high-quality anime faces using **Deep Convolutional Generative Adversarial Networks (DCGAN)**, trained on a curated anime dataset. Built with **PyTorch** for deep learning and **Streamlit** for an interactive user interface.

---

## 📸 Live Demo

> 🔗 Coming soon (Deploy on [Streamlit Cloud](https://streamlit.io/cloud) or [Hugging Face Spaces](https://huggingface.co/spaces))

---

## 🚀 Features

- 🎨 Generate realistic anime-style faces with one click
- 🎚️ Customize the number of faces and random seed
- ⚡ Fast inference with PyTorch
- 🌐 Responsive, professional UI using Streamlit
- 📦 Easy to extend with latent vector sliders and image-to-anime style conversion

---

## 📌 Tech Stack

| Component        | Technology            |
|------------------|------------------------|
| Deep Learning    | PyTorch, DCGAN         |
| Web UI           | Streamlit              |
| Data Processing  | torchvision, PIL       |
| Deployment Ready | Docker, Streamlit Cloud|

---

## 🧠 About the Model

This project uses a **DCGAN (Deep Convolutional GAN)** to learn the distribution of anime face images and generate new samples from a random noise vector (`z`). The generator progressively upsamples from a 100-dimensional latent space into 64x64 RGB images.

- Dataset: [Anime Face Dataset (Kaggle)](https://www.kaggle.com/datasets/splcher/animefacedataset)
- Input: 100-dim latent vector `z` sampled from a normal distribution
- Output: Synthetic anime faces in 64x64 resolution



## 📁 Project Structure

```

anime-dcgan/
├── models/                  # DCGAN architecture (Generator, Discriminator)
├── outputs/                 # Saved models and generated images
├── streamlit\_app/           # UI and backend logic
│   ├── app.py               # Streamlit app
│   └── generator\_utils.py   # PyTorch loading + generation
├── train.py                 # DCGAN training script
├── generate.py              # CLI face generator
├── README.md
└── .gitignore

````

---

## 🧪 How to Use

### 🔧 1. Install Dependencies

```bash
pip install -r requirements.txt
````

### 📦 2. Train the GAN (or use pretrained)

```bash
python train.py
```

Or download the pre-trained model:

```
outputs/netG_epoch_20.pth
```

### 🧪 3. Run the Generator Web App

```bash
cd streamlit_app
streamlit run app.py
```

---

## 📈 Results

The DCGAN generates anime faces with a high degree of realism after training for \~20 epochs. You can control the diversity of generated faces via:

* Latent vector (`z`)
* Random seed input
* Number of faces to sample

---



## 👨‍💻 Author

**Mohammad Hamim**,
BSc in Software Engineering, Zhengzhou University,
Researcher | Full-Stack & AI Developer,
[LinkedIn](https://linkedin.com/) | [GitHub](https://github.com/) | [Email](mailto:hamimmd555@gmail.com)

---

## 📝 License

MIT License. 
