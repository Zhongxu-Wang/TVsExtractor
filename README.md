# Acoustic-to-Articulatory Speech Inversion (AAI) Model

This model extracts physiological features from speech for subsequent speech tasks.  
For more information, please refer to the paper: [https://dl.acm.org/doi/10.1145/3664647.3681097](https://dl.acm.org/doi/10.1145/3664647.3681097)  
Project: [https://github.com/Zhongxu-Wang/ArtSpeech?tab=readme-ov-file](https://github.com/Zhongxu-Wang/ArtSpeech?tab=readme-ov-file)

## Inference (Extracting Physiological Features):

1. Download the pre-trained model: XXX, and place it in the `output/ckpt` directory.
2. Prepare your data files directory. For the specific format, refer to: `LibriTTS_dataset.txt` in the project.
3. Run `python synthesize.py`.  
   *(F0 extraction may take considerable time; you may consider performing this step in advance.)*

## Training:

The model is trained using the HPRC dataset and CleanUNet for denoising.

1. Download the processed dataset.
2. Run `python train.py`.

## Future Work:

1. The current TV data is a processed version of the EMA dataset, and there may be better methods for computing physiological features that can improve model accuracy and the efficiency of speech signal representation.
2. Providing textual information as additional input can improve extraction accuracy, but it will reduce the model's convenience.
