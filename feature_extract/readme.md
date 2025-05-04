# 📝 Feature Extraction Tips

This script utilizes a **frozen EAT model** to extract audio features. The extracted features are saved in `.npy` format at approximately 50Hz and support both **frame-level** and **utterance-level (CLS token)** representations.

## 🔧 Checkpoint Modes
This script support two types of checkpoints:

* **(EAT) Pre-trained:** Ideal for fine-tuning on various downstream tasks.
* **(EAT) Fine-tuned:** Recommended for direct feature extraction, often yielding better performance.


## 🔍 Feature Granularity

You can specify the granularity of extracted features depending on your target task:

| Granularity | Description                                  |
| ----------- | -------------------------------------------- |
| `all`       | Frame-level features **including** CLS token |
| `frame`     | Frame-level features **excluding** CLS token |
| `utterance` | Utterance-level embedding (CLS token only)   |

For **classification tasks**, utterance-level embeddings (CLS token) typically offer superior performance.


## 🎯 Target Length Configuration

The `target_length` parameter determines the length of the input mel-spectrogram (at 100Hz before downsampling). Recommendations:

* Use `1024` for 10-second clips
* Use `512` for 5-second clips

> ⚠️ **Note:** The EAT model was pre-trained on 10-second clips from AudioSet. For optimal results, pad or trim your audio to **10 seconds** and set `target_length = 1024`.

### ⚙️ CNN Encoder Constraints

The CNN encoder uses a 16×16 kernel with stride 16. To ensure valid convolution, **make sure `target_length` is a multiple of 16**.

Here is an example snippet to adjust mel-spectrogram length accordingly: 

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
