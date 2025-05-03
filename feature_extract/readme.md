## 📝 Feature Extraction Tips
- This script uses a **frozen EAT model** to extract audio features. You can adapt it for fine-tuning as needed. 
- For **classification tasks**, utterance-level features generally yield better results than frame-level features. 
- The `target_length` parameter controls mel-spectrogram length. At 100Hz, we recommend:
  - `1024` for 10-second clips
  - `512` for 5-second clips
- ⚠️ EAT was pretrained on 10-second AudioSet clips. For best performance, **pad or trim your audio to 10s** and set `target_length = 1024`.
- The CNN encoder uses a 16×16 kernel with stride 16. To ensure valid convolution, **make sure `target_length` is a multiple of 16**.
- Example code to align mel-spectrogram length: 

```python
n_frames = source.shape[1]
target_length = ((n_frames + 15) // 16) * 16  # Round up to nearest multiple of 16
diff = target_length - n_frames
if diff > 0:
    m = torch.nn.ZeroPad2d((0, 0, 0, diff)) 
    source = m(source)
elif diff < 0:
    source = source[:, :target_length, :]
```
