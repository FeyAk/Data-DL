Penguins and UKM Dataset Machine Learning & Visualization

This repository contains machine learning experiments, data visualization, and neural network modeling using Python, including decision tree visualization (dtreeviz), K-nearest neighbors (KNN), and a small TensorFlow example. It covers two datasets: the penguins dataset (from Seaborn) and the UCI "User Knowledge Modeling" (UKM) dataset.

Datasets:
Penguins
- Features: bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g
- Target: species (Adelie, Chinstrap, Gentoo)

UCI UKM
- Features: STG, SCG, STR, LPR, PEG
- Target: UNS (Very Low, Low, Middle, High)

Setup
  
    git clone https://github.com/yourusername/your-repo.git
  
    cd your-repo
  
    conda create -n ml_project python=3.11
  
    conda activate ml_project
  
    pip install numpy pandas seaborn matplotlib scikit-learn dtreeviz tensorflow keras ucimlrepo openpyxl

Note: Graphviz required for dtreeviz and plot_model. Check with dot -V.


Workflows:
Penguins Dataset
- Clean data (drop unused columns, remove missing values).Visualize pairwise features: sns.pairplot(df, hue="species")
- KNN Classification → ~88% accuracy
- Decision Tree → ~97% accuracy, visualize with dtreeviz

UCI UKM Dataset
- Fetch via ucimlrepo
- Join features and target
- Optional Excel export: df.to_excel("uci_ukm.xlsx")

TensorFlow Example
- Small binary classification dataset
- Sequential model: Dense(8, ReLU, HeNormal) → Dense(1, Sigmoid)
- Optimizer: SGD (0.1), Loss: binary_crossentropy
- Training plotted with matplotlib, predictions via model.predict()

License:
MIT License
