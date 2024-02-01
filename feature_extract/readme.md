## Tips :sparkles:
- This script demonstrates using frozen EAT to extract audio features , you could also modify the codes to fine-tune an end-to-end model. 
- For **classification** tasks, using utterance-level features often performs **better** than using frame-level features (with mean pooling methods). 
- `target_length` is the padding parameter. Since we utilize 100Hz fbank features, we recommand `target_length = 1024` for 10s audio clips and `target_length = 512` for 5s audio clips. You could also adjust the codes with your dataset.  