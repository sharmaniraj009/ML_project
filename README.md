# Pneumonia Detection using Deep Learning

## Overview
This project aims to detect pneumonia from chest X-ray images using deep learning techniques. It employs convolutional neural networks (CNNs) to classify X-ray images as either pneumonia-positive or normal.

## Dataset
The dataset used in this project is derived from publicly available chest X-ray datasets, containing labeled images of normal and pneumonia-affected lungs.

## Requirements
To run this project, ensure you have the following dependencies installed:
- Python 3.11
- TensorFlow/Keras
- NumPy
- Matplotlib
- OpenCV
- Scikit-learn
- Pandas

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Implementation
The project follows these key steps:
1. **Data Preprocessing**: Image resizing, normalization, and augmentation.
2. **Model Training**: Using a CNN architecture for classification.
3. **Evaluation**: Assessing model performance using accuracy, precision, recall, and F1-score.
4. **Prediction**: Making predictions on new X-ray images.

## Usage
Run the Jupyter Notebook to train and evaluate the model:
```bash
jupyter notebook pneumonia-detection.ipynb
```

## Results
The trained model achieves high accuracy in detecting pneumonia from chest X-rays. Evaluation metrics and sample predictions are displayed in the notebook.

## Future Work
- Improve accuracy with advanced CNN architectures.
- Deploy as a web-based tool for medical diagnostics.

## License
This project is licensed under the MIT License.

## Author
[Niraj Sharman](https://github.com/sharmaniraj009)

