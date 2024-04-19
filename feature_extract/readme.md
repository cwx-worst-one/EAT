## Tips :sparkles:
- This script demonstrates using **frozen** EAT to extract audio features , you could also modify the codes to fine-tune an end-to-end model. 
- For **classification** tasks, using utterance-level features often performs **better** than using frame-level features (with mean pooling methods). 
- `target_length` is the padding parameter. Since we use 100Hz fbank features, we recommend setting `target_length = 1024` for 10-second audio clips and `target_length = 512` for 5-second audio clips. **Notably**, as the EAT model was pretrained on the 10-second Audioset dataset, it excels in extracting features from 10-second audio segments. Therefore, you might consider **padding or trimming** your audio to 10 seconds (i.e., to set `target_length = 1024`) to optimize feature extraction using the EAT model.
- You could also adjust the codes with your dataset. **Importantly**, as our CNN encoder utilizes a $16×16$ convolution kernel with a stride of 16 for feature extraction from the Mel spectrogram, to avoid overlap, it's best to ensure that the target_length is a multiple of 16. An example is as below:  
    ```py
    ...
    source = torchaudio.compliance.kaldi.fbank(source, htk_compat=True, sample_frequency=16000, use_energy=False,
    window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10).unsqueeze(dim=0)
    n_frames = source.shape[1]
    target_length = n_frames
    if target_length % 16 != 0:
        target_length = n_frames + (16 - n_frames % 16)
    diff = target_length - n_frames
    if diff > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, diff)) 
        source = m(source)
    elif diff < 0:
        source = source[:,:target_length, :]
    ...
    ```