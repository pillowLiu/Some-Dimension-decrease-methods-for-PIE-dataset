Face Recognition Project
-----------------------------------------------------------------------------------

## Overview
--------
This project uses PCA, LDA, GMM, and SVM for face recognition. It's all about reducing dimensions and classifying faces in images.

## Contents

PCA_stuff.py: Reduces image dimensions with PCA and uses K-Nearest Neighbors for classification. Includes 2D and 3D visualizations.
LDA_magic.py: Applies LDA for dimension reduction and classification. Visualizes results in 2D and 3D.
GMM_clusters.py: Clusters face images using GMM. Shows how images are grouped.
SVM_classifier.py: Uses SVM for face classification. Experiments with different parameters for accuracy.

## Running the Code
Ensure Python is installed. In the terminal, navigate to the project folder and run:

bash
Copy code
python PCA_stuff.py
python LDA_magic.py
python GMM_clusters.py
python SVM_classifier.py

This will execute the scripts and display results.

## Dependencies
Install matplotlib, numpy, sklearn, and PIL with:

bash
Copy code
pip install matplotlib numpy sklearn pillow
Face images should be in a PIE folder, with numbered subfolders for each person.

## Contact
If you have any questions, feel free to reach out to me at e0954758@u.nus.edu.
