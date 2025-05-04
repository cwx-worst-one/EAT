import argparse
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
import csv
from dataclasses import dataclass
from transformers import AutoModel

# ===== Argument Parser =====
def get_parser():
    parser = argparse.ArgumentParser(description="Use fine-tuned EAT for acoustic event classification")
    parser.add_argument('--source_file', required=True, help='Path to input .wav file')
    parser.add_argument('--label_file', required=True, help='Path to label CSV file')
    parser.add_argument('--model_dir', required=True, help='Model definition directory (not needed for HF framework)')
    parser.add_argument('--checkpoint_dir', required=True, help='Checkpoint path or HF model ID')
    parser.add_argument('--target_length', type=int, required=True, help='Target mel length (time dimension)')
    parser.add_argument('--top_k_prediction', type=int, required=True, help='Top-k predicted labels')
    parser.add_argument('--norm_mean', type=float, default=-4.268, help='Normalization mean')
    parser.add_argument('--norm_std', type=float, default=4.569, help='Normalization std')
    parser.add_argument('--framework', required=True, choices=['fairseq', 'huggingface'], help='Model framework')
    return parser


# ===== Label Loader =====
def build_dictionary(label_path):
    vocab = {}
    with open(label_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            index, label = int(row[0]), row[2]
            vocab[index] = label
    return vocab


# ===== Fairseq Helper =====
@dataclass
class UserDirModule:
    user_dir: str


# ===== Model Loader =====
def load_model(args):
    if args.framework == 'huggingface':
        model = AutoModel.from_pretrained(args.checkpoint_dir, trust_remote_code=True).eval().cuda()
    elif args.framework == 'fairseq':
        import fairseq
        model_path = UserDirModule(args.model_dir)
        fairseq.utils.import_user_module(model_path)
        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([args.checkpoint_dir])
        model = model[0].eval().cuda()
    else:
        raise ValueError(f"Unsupported framework: {args.framework}")
    return model


# ===== Audio Preprocessing =====
def preprocess_audio(path, target_length, norm_mean, norm_std):
    assert path.endswith('.wav'), "Input must be a .wav file"
    wav, sr = sf.read(path)
    assert sf.info(path).channels == 1, f"Expected mono audio, got {sf.info(path).channels}"

    wav = torch.tensor(wav).float()
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = wav - wav.mean()

    mel = torchaudio.compliance.kaldi.fbank(
        wav.unsqueeze(0), htk_compat=True, sample_frequency=16000,
        use_energy=False, window_type='hanning', num_mel_bins=128,
        dither=0.0, frame_shift=10
    ).unsqueeze(0)

    # pad or truncate
    n_frames = mel.shape[1]
    if n_frames < target_length:
        mel = torch.nn.ZeroPad2d((0, 0, 0, target_length - n_frames))(mel)
    elif n_frames > target_length:
        mel = mel[:, :target_length, :]

    mel = (mel - norm_mean) / (norm_std * 2)
    return mel.unsqueeze(0).cuda()  # shape [1, 1, T, F]


# ===== Main Inference =====
def main():
    args = get_parser().parse_args()

    model = load_model(args)
    vocab = build_dictionary(args.label_file)
    mel = preprocess_audio(args.source_file, args.target_length, args.norm_mean, args.norm_std)

    with torch.no_grad():
        try:
            logits = model(mel)
            probs = torch.sigmoid(logits)
            values, indices = torch.topk(probs, args.top_k_prediction)

            print("\n************ Acoustic Event Inference ************")
            print("LABEL".ljust(30) + "PREDICTION")
            for i, v in zip(indices[0], values[0]):
                print(f"{vocab[i.item()]:<30} {v.item():.3f}")
            print("**************************************************\n")
        except Exception as e:
            print(f"Inference failed: {e}")
            raise


if __name__ == '__main__':
    main()
