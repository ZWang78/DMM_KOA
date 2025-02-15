# Temporal Evolution of Knee Osteoarthritis: A Diffusion-based Morphing Model for X-Ray Medical Image Synthesis

This repository contains a Diffusion-based Morphing Model (DMM) for generating anatomically accurate Knee Osteoarthritis (KOA) X-ray image sequences.

## 🚀 Features
- Uses **Denoising Diffusion Probabilistic Model (DDPM)** for KOA image synthesis.
- Incorporates a **registration-based morphing module** to preserve anatomical topology.
- Performs **cross-database validation** using OAI and MOST datasets.

## 🔬 Methodology
1. **Diffusion Model for Image Generation**  
   - Trains a **Denoising Diffusion Probabilistic Model (DDPM)** to reconstruct knee X-ray images at different KL grades.  
   - The model gradually removes noise to generate high-quality KOA progression sequences.

2. **Morphing Module for Temporal Evolution**  
   - A **spatial transformer network (STN)** aligns X-ray images across different KOA stages.  
   - The **deformation field** preserves local anatomical structures while simulating disease progression.  

3. **Cross-Database Validation (OAI & MOST)**  
   - The model is trained and evaluated on two major knee osteoarthritis datasets:  
     - **OAI (Osteoarthritis Initiative)**  
     - **MOST (Multicenter Osteoarthritis Study)**  
   - Ensures robustness and generalization across different imaging sources.

## 📊 Training & Inference
To train the model, run:
```bash
python train.py --dataset_path ./OAI_m
python inference.py --image_path ./test_xray.png


