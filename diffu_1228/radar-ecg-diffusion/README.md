# Radar-ECG Diffusion Model

This project implements a diffusion model to reconstruct electrocardiogram (ECG) data from millimeter-wave radar data. The model leverages a conditional U-Net architecture and employs a diffusion process for generating high-quality ECG signals.

## Project Structure

```
radar-ecg-diffusion
├── src
│   ├── data
│   │   ├── __init__.py          # Initializes the data module
│   │   ├── dataset.py           # Defines the RadarECGDataset class for loading and processing data
│   │   └── preprocessing.py      # Contains data preprocessing functions
│   ├── models
│   │   ├── __init__.py          # Initializes the model module
│   │   ├── unet.py              # Defines the ConditionalUNet class for ECG reconstruction
│   │   ├── scheduler.py          # Manages the diffusion process
│   │   └── embeddings.py         # Generates sinusoidal position embeddings
│   ├── training
│   │   ├── __init__.py          # Initializes the training module
│   │   ├── trainer.py            # Contains the training loop
│   │   └── losses.py             # Defines loss functions
│   ├── inference
│   │   ├── __init__.py          # Initializes the inference module
│   │   └── sampler.py            # Implements the DDIM sampling function
│   ├── evaluation
│   │   ├── __init__.py          # Initializes the evaluation module
│   │   ├── metrics.py            # Contains evaluation metrics calculations
│   │   └── visualization.py       # Includes visualization functions for results comparison
│   ├── utils
│   │   ├── __init__.py          # Initializes the utility module
│   │   └── helpers.py            # Contains helper functions
│   └── main.py                   # Entry point for the project
├── configs
│   └── default.yaml              # Configuration file for model and training parameters
├── scripts
│   ├── train.py                  # Script to start the training process
│   └── evaluate.py               # Script to start the evaluation process
├── tests
│   └── test_model.py             # Unit tests for the model
├── requirements.txt              # Lists required Python packages
├── setup.py                      # Setup script for packaging and installation
└── README.md                     # Project documentation and usage instructions
```

## Installation

To install the required dependencies, run:

```
pip install -r requirements.txt
```

## Usage

1. **Training the Model**: To train the model, use the following command:

   ```
   python scripts/train.py --data <path_to_your_dataset>
   ```

2. **Evaluating the Model**: To evaluate the trained model, run:

   ```
   python scripts/evaluate.py --model <path_to_your_model>
   ```

## License

This project is licensed under the MIT License. See the LICENSE file for details.