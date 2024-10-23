# AdaptMoist: Texture-Driven Domain Adaptation for Moisture Content Prediction in Wood Chips

**AdaptMoist** is a domain adaptation model designed to predict the moisture content of wood chips using texture-based features extracted from RGB images. This repository contains the implementation of AdaptMoist, which is based on the Domain-Adversarial Neural Network (DANN) framework, along with additional callback based on Adjusted Mutual Information (AMI).


## Features

- **Feature Extraction:** Five different types of texture features extracted from RGB images.
- **Comprehensive Evaluation:** Extensive evaluation of machine learning predictors on texture features for moisture prediction
- **Domain Adaptation:** DANN-based architecture with AMI-based callback
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score.


## Installation

To set up the environment and install dependencies, follow the steps below:

1. Clone this repository:

    ```bash
    git clone https://github.com/abdurrahman1828/adaptmoist.git
    cd adaptmoist
    ```

2. Create a virtual environment (optional but recommended):

    ```bash
    python -m venv env
    source env/bin/activate  # For Linux/macOS
    env\Scripts\activate     # For Windows
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Data Preparation

Before running the model, make sure you have the dataset of wood chip images **OR** the corresponding extracted texture features in CSV format. You can extract the features from RGB images using ```extract_[feature_name].py``` scripts.

### Running the comprehensive evaluation

You can run the evaluation on 12 models using ```ml_models_cross_validate.py``` script. You may need to provide the required directories for data and results. 

### Running the Model

To run the domain adaptation model, use the ```adaptmoist.py``` script.


## Acknowledgments

This project was made possible with the help of several open-source libraries and repositories. We would like to acknowledge the following GitHub projects:

- [adapt](https://github.com/adapt-python/adapt): For providing the core domain adaptation methods such as DANN, MDD, and CDAN.
- [scikit-learn](https://github.com/scikit-learn/scikit-learn): For machine learning utilities and models, including classification algorithms, clustering, and evaluation metrics.
- [pyfeats](https://github.com/giakoumoglou/pyfeats/tree/main): For image feature extraction.


## Contact

If you have any questions or suggestions regarding this project, feel free to reach out:

- Email: [ar2806@msstate.edu](mailto:ar2806@msstate.edu)


