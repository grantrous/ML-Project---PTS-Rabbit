# Machine Learning Model for Predicting Rabbit Positions Based on Pressure Data

This project aims to develop a machine learning model using PyTorch to predict the position of a rabbit based on various pressure readings. The model will take inputs from six different pressure categories and classify the rabbit's position into one of three classes.

## Project Structure

- **src/**: Contains the main source code for data processing, model definition, training, evaluation, and utility functions.
  - **data_preprocessing.py**: Functions for loading and preprocessing the dataset, including reading CSV files, normalizing pressure data, and encoding rabbit position classes.
  - **model.py**: Defines the `PressureRabbitModel` class, which inherits from `nn.Module`. This class includes methods for initializing the model architecture and defining the forward pass.
  - **train.py**: Responsible for training the model, including setting up the training loop, calculating loss, and updating model weights. It also handles dataset splitting into training and validation sets.
  - **evaluate.py**: Contains functions for evaluating the model's performance on the test set, calculating metrics such as accuracy and loss, and visualizing predictions versus true labels.
  - **utils.py**: Includes utility functions for data visualization, saving and loading model weights, and other helper functions needed throughout the project.

- **requirements.txt**: Lists the Python packages required for the project, including PyTorch, pandas, numpy, and other dependencies necessary for data processing and model training.

- **README.md**: This documentation file provides an overview of the project, setup instructions, usage examples, and other relevant information.

- **Work With Raw Data.ipynb**: A Jupyter notebook for exploratory data analysis and initial data processing steps, including visualizations and preliminary model training experiments.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd ml-pressure-rabbit
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Prepare your dataset in the expected format and place it in the appropriate directory.

## Usage

1. Preprocess the data:
   Run the `data_preprocessing.py` script to load and preprocess your dataset.

2. Train the model:
   Use the `train.py` script to train the model on the preprocessed data.

3. Evaluate the model:
   After training, run the `evaluate.py` script to assess the model's performance on the test set.

## Example

To train the model, you can run:
```
python src/train.py
```

To evaluate the model, use:
```
python src/evaluate.py
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.