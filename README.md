# Airline Satisfaction Deployment

## Project Overview

This project focuses on deploying machine learning models to predict airline passenger satisfaction. It includes the necessary files and instructions to set up and run the deployment environment, enabling real-time predictions based on input features.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Model Details](#model-details)
4. [Contributing](#contributing)

## Installation

To set up this project locally, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Ziad-Shalaby/Airline_Satisfaction-Deployment.git
   ```

2. **Navigate to the project directory:**

   ```bash
   cd Airline_Satisfaction-Deployment
   ```

3. **Create a virtual environment:**

   ```bash
   python -m venv env
   ```

4. **Activate the virtual environment:**

   - On Windows:

     ```bash
     env\Scripts\activate
     ```

   - On macOS and Linux:

     ```bash
     source env/bin/activate
     ```

5. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

After installing the dependencies, you can run the deployment script as follows:

1. **Start the application:**

   ```bash
   python air.py
   ```

2. **Access the application:**

   Open your web browser and navigate to `http://localhost:5000` to interact with the deployment interface.

3. **Run it online:**

   You can also access the application online at [Airline Satisfaction Deployment](https://airlinesatisfaction-deployment.streamlit.app/).

## Model Details

The deployment includes several pre-trained machine learning models stored as `.pkl` files:

- `GradientBoostingClassifier.pkl`
- `Naive.pkl`
- `decision_tree.pkl`
- `knn.pkl`
- `svm.pkl`

These models are used to predict passenger satisfaction based on input features. Additionally, encoder and scaler files are provided to preprocess input data appropriately.

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. **Fork the repository.**
2. **Create a new branch:**

   ```bash
   git checkout -b feature/YourFeatureName
   ```

3. **Make your changes and commit them:**

   ```bash
   git commit -m 'Add some feature'
   ```

4. **Push to the branch:**

   ```bash
   git push origin feature/YourFeatureName
   ```

5. **Open a pull request.**

---

Thank you for checking out the Airline Satisfaction Deployment project! We hope you find it helpful and easy to use.
