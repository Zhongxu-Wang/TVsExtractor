<div align="center">
<h1>
<b>
Acoustic-to-Articulatory Speech Inversion (AAI) Model
</b>
</h1>
</div>



This model extracts articulatory features from speech for subsequent speech tasks.  
For more information, please refer to the paper: <a href="https://dl.acm.org/doi/10.1145/3664647.3681097" target="_blank">Paper</a> and the
Project: <a href="https://github.com/Zhongxu-Wang/ArtSpeech?tab=readme-ov-file" target="_blank">ArtSpeech</a>

## Inference (Extracting Articulatory Features):

1. Download the <a href="https://drive.google.com/file/d/1wxs1OoBsTTRMuMP2OQ6f2m9WpgdQIkIB/view?usp=drive_link" target="_blank">pre-trained model</a>, and place it in the `output/ckpt` directory.
2. Prepare your data files directory. For the specific format, refer to: `LibriTTS_dataset.txt` in the project.
3. Run `python synthesize.py`.  
   *(F0 extraction may take considerable time; you may consider performing this step in advance.)*

## Training:

The model is trained using the HPRC dataset and we have preprocessed the speech data in the HPRC dataset by applying <a href="https://github.com/NVIDIA/CleanUNet" target="_blank">CleanUNet</a> for denoising.

1. Download the <a href="https://drive.google.com/file/d/1iS99bw2p97bWTo_frf2wi1LUiX3F5ieJ/view?usp=drive_link" target="_blank">processed dataset</a>.
2. Run `python train.py`.

## Future Work:

1. The current TVs is a processed version of the EMA dataset, and there may be better methods for computing articulatory features that can improve model accuracy and the efficiency of speech signal representation.
2. Providing textual information as additional input can improve extraction accuracy, but it will reduce the model's convenience.
