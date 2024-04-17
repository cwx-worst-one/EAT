import argparse
from dataclasses import dataclass
import numpy as np
import soundfile as sf

import torch
import torch.nn.functional as F
import fairseq
import torchaudio

# global normalization for AudioSet as default (norm_mean=-4.268 && norm_std=4.569)
def get_parser():
    parser = argparse.ArgumentParser(
        description="extract EAT features for downstream tasks"
    )
    parser.add_argument('--source_file', help='location of source wav files', required=True)
    parser.add_argument('--target_file', help='location of target npy files', required=True)
    parser.add_argument('--model_dir', type=str, help='pretrained model', required=True)
    parser.add_argument('--checkpoint_dir', type=str, help='checkpoint for pre-trained model', required=True)
    parser.add_argument('--granularity', type=str, choices=['all', 'frame', 'utterance'], required=True, 
                        help='Specifies the granularity of features to use: "all" for all frame features including the cls token, \
                        "frame" for all frame features excluding the cls token, and "utterance" for only the cls token feature.')
    parser.add_argument('--target_length', type=int, help='the target length of Mel spectrogram in time dimension', required=True)
    parser.add_argument('--norm_mean', type=float, help='mean value for normalization', default=-4.268)
    parser.add_argument('--norm_std', type=float, help='standard deviation for normalization', default=4.569)
    parser.add_argument('--mode', type=str, choices=['pretrain', 'finetune'], required=True, 
                        help='Specifies the mode of the model: "pretrain" for EAT pre-training, "finetune" for EAT fine-tuning.')
    return parser

@dataclass
class UserDirModule:
    user_dir: str

def main():
    parser = get_parser()
    args = parser.parse_args()
    print(args)

    source_file = args.source_file
    target_file = args.target_file
    model_dir = args.model_dir
    checkpoint_dir = args.checkpoint_dir
    granularity = args.granularity
    target_length = args.target_length
    norm_mean = args.norm_mean
    norm_std = args.norm_std
    mode = args.mode

    model_path = UserDirModule(model_dir)
    fairseq.utils.import_user_module(model_path)
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint_dir])
    model = model[0]
    model.eval()
    model.cuda()

    assert source_file.endswith('.wav'), "the standard format of file should be '.wav' "
    wav, sr = sf.read(source_file)
    channel = sf.info(source_file).channels
    source = torch.from_numpy(wav).float().cuda()
    if sr == 16e3:
        print("Original sample rate is already 16kHz in file {}".format(source_file))
    else: 
        source = torchaudio.functional.resample(source, orig_freq=sr, new_freq=16000).float().cuda()
        print("It is resampled to 16kHz in file {}".format(source_file))
        
    assert channel == 1, "Channel should be 1, but got {} in file {}".format(channel, source_file)
    
    source = source - source.mean()
    source = source.unsqueeze(dim=0)
    source = torchaudio.compliance.kaldi.fbank(source, htk_compat=True, sample_frequency=16000, use_energy=False,
        window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10).unsqueeze(dim=0)
    
    n_frames = source.shape[1]
    diff = target_length - n_frames
    if diff > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, diff)) 
        source = m(source)
        
    elif diff < 0:
        source = source[:,0:target_length, :]
                
    source = (source - norm_mean) / (norm_std * 2)
    
    if mode == "finetune":
        model = model.model
        
    with torch.no_grad():
        try:
            source = source.unsqueeze(dim=0) #btz=1
            
            if granularity == 'all':
                feats = model.extract_features(source, padding_mask=None,mask=False, remove_extra_tokens=False)
                feats = feats['x'].squeeze(0).cpu().numpy()
            elif granularity == 'frame':
                feats = model.extract_features(source, padding_mask=None,mask=False, remove_extra_tokens=True)
                feats = feats['x'].squeeze(0).cpu().numpy()
            elif granularity == 'utterance':
                feats = model.extract_features(source, padding_mask=None,mask=False, remove_extra_tokens=False)
                feats = feats['x']
                feats = feats[:, 0].squeeze(0).cpu().numpy()
            else:
                raise ValueError("Unknown granularity: {}".format(args.granularity))
            print(feats.shape)
            np.save(target_file, feats)
            print("Successfully saved")
        except:
            print("Error in extracting features from {}".format(source_file))
            Exception("Error in extracting features from {}".format(source_file))


if __name__ == '__main__':
    main()