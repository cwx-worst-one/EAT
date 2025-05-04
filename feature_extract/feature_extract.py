import argparse
from dataclasses import dataclass
import numpy as np
import soundfile as sf

import torch
import torchaudio


# ===== Argument Parser =====
def get_parser():
    parser = argparse.ArgumentParser(description="Extract EAT features for downstream tasks")
    parser.add_argument('--source_file', required=True, help='Path to input .wav file')
    parser.add_argument('--target_file', required=True, help='Path to output .npy file')
    parser.add_argument('--model_dir', required=True, help='Directory containing the model definition (not needed for HF framework)')
    parser.add_argument('--checkpoint_dir', required=True, help='Checkpoint path or HF model ID')
    parser.add_argument('--granularity', required=True, choices=['all', 'frame', 'utterance'],
                        help='Feature type: "all" (including CLS), "frame" (excluding CLS), or "utterance" (CLS only)')
    parser.add_argument('--target_length', required=True, type=int, help='Target length of mel-spectrogram')
    parser.add_argument('--norm_mean', type=float, default=-4.268, help='Normalization mean')
    parser.add_argument('--norm_std', type=float, default=4.569, help='Normalization std')
    parser.add_argument('--mode', required=True, choices=['pretrain', 'finetune'], help='Model mode')
    parser.add_argument('--framework', required=True, choices=['fairseq', 'huggingface'], help='Framework to use')
    return parser

@dataclass
class UserDirModule:
    user_dir: str

# ===== Model Loader =====
def load_model(args):
    if args.framework == "huggingface":
        from transformers import AutoModel
        model = AutoModel.from_pretrained(args.checkpoint_dir, trust_remote_code=True).eval().cuda()
    elif args.framework == "fairseq":
        import fairseq
        model_path = UserDirModule(args.model_dir)
        fairseq.utils.import_user_module(model_path)
        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([args.checkpoint_dir])
        model = models[0]
        if args.mode == "finetune":
            model = model.model
        model.eval().cuda()
    else:
        raise ValueError(f"Unsupported framework: {args.framework}")
    return model


# ===== Feature Extraction For Two Frameworks =====
def extract_feature_tensor(model, x, framework):
    if framework == "huggingface":
        return model.extract_features(x)
    elif framework == "fairseq":
        return model.extract_features(x, padding_mask=None, mask=False, remove_extra_tokens=False)['x']
    else:
        raise ValueError(f"Unsupported framework: {framework}")


# ===== Feature Extraction Pipeline =====
def extract_features(args):
    assert args.source_file.endswith('.wav'), "Source file must be a .wav file"

    # Load waveform and resample to 16kHz if necessary
    wav, sr = sf.read(args.source_file)
    assert sf.info(args.source_file).channels == 1, "Only mono-channel audio is supported"
    waveform = torch.tensor(wav).float().cuda()
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        print(f"Resampled to 16kHz: {args.source_file}")

    # Normalize and convert to mel-spectrogram
    waveform = waveform - waveform.mean()
    mel = torchaudio.compliance.kaldi.fbank(
        waveform.unsqueeze(0),
        htk_compat=True,
        sample_frequency=16000,
        use_energy=False,
        window_type='hanning',
        num_mel_bins=128,
        dither=0.0,
        frame_shift=10
    ).unsqueeze(0)

    # Pad or truncate to target length
    n_frames = mel.shape[1]
    if n_frames < args.target_length:
        mel = torch.nn.ZeroPad2d((0, 0, 0, args.target_length - n_frames))(mel)
    elif n_frames > args.target_length:
        mel = mel[:, :args.target_length, :]

    mel = (mel - args.norm_mean) / (args.norm_std * 2)
    mel = mel.unsqueeze(0).cuda()  # shape: [1, 1, T, F]

    model = load_model(args)

    with torch.no_grad():
        try:
            result = extract_feature_tensor(model, mel, args.framework)

            if args.granularity == 'frame':
                result = result[:, 1:, :]     # remove CLS token
            elif args.granularity == 'utterance':
                result = result[:, 0]         # keep only CLS token

            result = result.squeeze(0).cpu().numpy()
            np.save(args.target_file, result)
            print(f"Feature shape: {result.shape}")
            print(f"Saved to: {args.target_file}")

        except Exception as e:
            print(f"Feature extraction failed: {e}")
            raise

# ===== Entry =====
if __name__ == '__main__':
    args = get_parser().parse_args()
    extract_features(args)