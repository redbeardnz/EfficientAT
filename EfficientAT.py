import torch
import librosa
import numpy as np
from pathlib import Path
#from scipy.signal.windows import gaussian as gaussian_window
from scipy.stats import norm
from torch import autocast
from contextlib import nullcontext, suppress

from models.mn.model import get_model as get_mobilenet
from models.dymn.model import get_model as get_dymn
from models.ensemble import get_ensemble_model
from models.preprocess import AugmentMelSTFT
from helpers.utils import NAME_TO_WIDTH, labels


class EfficientAT:

    def __init__(self,
                 model_name: str = "mn10_as",
                 strides: list = [2, 2, 2, 2],
                 head_type: str = "mlp",
                 cuda: bool = True,
    ):
        if model_name.startswith("dymn"):
            self._model = get_dymn(width_mult=NAME_TO_WIDTH(model_name),
                                   pretrained_name=model_name,
                                   strides=strides)
        else:
            self._model = get_mobilenet(width_mult=NAME_TO_WIDTH(model_name),
                                       pretrained_name=model_name,
                                       strides=strides,
                                       head_type=head_type)
        self._device = torch.device('cuda') if cuda and torch.cuda.is_available() else torch.device('cpu')


    def predict(self,
                wav: Path,
                resolution: int = 1,
                pad_method: str = "repeat",
                top_rank: int = 10,
                **kwargs,
    ):
        self._model.to(self._device)
        self._model.eval()

        # model to preprocess waveform into mel spectrograms
        settings = {"n_mels": 128, "sr": 32000, "win_length": 800, "hopsize": 320}
        settings.update(kwargs)
        mel = AugmentMelSTFT(**settings)
        mel.to(self._device)
        mel.eval()

        sample_rate = settings["sr"]
        model_wav_length = 10 # in seconds
        seg_length = resolution # in seconds
        seg_size = sample_rate * seg_length # frames number of each seg
        waveform, _ = librosa.core.load(str(wav), sr=sample_rate, mono=True)
        results = list()

        for seg_id in range(int(waveform.size / seg_size)):
            seg = waveform[seg_id*seg_size : (seg_id+1)*seg_size]
            if pad_method == "repeat":
                # repeat seg to a 10 seconds audio
                wave = np.tile(seg, int(model_wav_length/seg_length))
            else:
                # pad the clip symmetrically with zeros on either side
                wave_size = model_wav_length * sample_rate
                wave = np.zeros(wave_size, dtype=np.float32)
                wave[int(wave_size/2)-int(seg_size/2) : int(wave_size/2)+int(seg_size/2)] = seg

            wave = torch.from_numpy(wave[None, :]).to(self._device)

            # our models are trained in half precision mode (torch.float16)
            # run on cuda with torch.float16 to get the best performance
            # running on cpu with torch.float32 gives similar performance,
            # using torch.bfloat16 is worse
            with torch.no_grad(), autocast(device_type=self._device.type) if self._device.type == 'cuda' else nullcontext():
                spec = mel(wave)
                preds, features = self._model(spec.unsqueeze(0))
            preds = torch.sigmoid(preds.float()).squeeze().cpu().numpy()

            sorted_indexes = np.argsort(preds)[::-1]

            result = dict()
            for k in range(top_rank if top_rank > 0 else 10):
                result[labels[sorted_indexes[k]]] = preds[sorted_indexes[k]]
            results.append(result)
        return results


    def predictGW(self, wav: Path,
                  resolution: int = 1,
                  sigma: float = 0.5,
                  top_rank: int = 10,
                  **kwargs):
        self._model.to(self._device)
        self._model.eval()

        # model to preprocess waveform into mel spectrograms
        settings = {"n_mels": 128, "sr": 32000, "win_length": 800, "hopsize": 320}
        settings.update(kwargs)
        mel = AugmentMelSTFT(**settings)
        mel.to(self._device)
        mel.eval()

        sample_rate = settings["sr"]
        step_dur = resolution # in seconds
        step_size = step_dur * sample_rate
        gaussian_window_dur = 10 # in seconds
        gaussian_window_size = gaussian_window_dur * sample_rate
        gaussian_window_step_size = int(gaussian_window_dur / step_dur)
        gwindow = self._gaussian_window(gaussian_window_step_size, scale=sigma)
        symme_pad_dur = gaussian_window_dur - step_dur
        symme_pad_step_size = int(symme_pad_dur / step_dur)
        symme_pad = np.zeros(symme_pad_dur * sample_rate, dtype=np.float32)
        waveform, _ = librosa.core.load(str(wav), sr=sample_rate, mono=True)
        tail_pad_size = (step_size - waveform.size % step_size) % step_size
        tail_pad = np.zeros(tail_pad_size, dtype=np.float32)
        waveform = np.concatenate((symme_pad, waveform, tail_pad, symme_pad))
        total_step_count = int((waveform.size - (gaussian_window_dur - step_dur) * sample_rate) / (sample_rate * step_dur))
        step_results = [dict() for i in range(int(int(waveform.size / sample_rate) / step_dur))]
        filter_out_labels = ["Silence", "Speech"]
        top_rank = top_rank if top_rank > 0 else 10

        #import pdb; pdb.set_trace()
        for step in range(total_step_count):
            #print(f"step: {step}")
            wave = waveform[step * step_size : step * step_size + gaussian_window_size]
            wave = torch.from_numpy(wave[None, :]).to(self._device)

            # our models are trained in half precision mode (torch.float16)
            # run on cuda with torch.float16 to get the best performance
            # running on cpu with torch.float32 gives similar performance,
            # using torch.bfloat16 is worse
            with torch.no_grad(), autocast(device_type=self._device.type) if self._device.type == 'cuda' else nullcontext():
                spec = mel(wave)
                preds, features = self._model(spec.unsqueeze(0))
            preds = torch.sigmoid(preds.float()).squeeze().cpu().numpy()

            sorted_indexes = np.argsort(preds)[::-1]

            for k in range(top_rank):
                label = labels[sorted_indexes[k]]
                predict = preds[sorted_indexes[k]]
                #print('{}: {:.3f}'.format(label, predict))
                if label in filter_out_labels:
                    continue
                for i, step_result in enumerate(step_results[step : step + gaussian_window_step_size]):
                    step_result[label] = step_result.get(label, 0.0) + predict * gwindow[i]

        results = list()
        step_results = step_results[symme_pad_step_size : len(step_results) - symme_pad_step_size]
        for i, step_result in enumerate(step_results):
            result = dict()
            count = 1
            for label, predict in sorted(step_result.items(), key=lambda x:x[1], reverse=True):
                if count > top_rank:
                    break
                result[label] = predict
                count += 1
            results.append(result)
        return results


    def _gaussian_window(self, points_num: int, scale: float):
        n = norm(loc=points_num/2, scale=scale)
        dists = list()
        for i in range(points_num):
            h = n.cdf(i+1)
            l = n.cdf(i)
            if i == 0:
                l = 0
            elif i == (points_num - 1):
                h = 1
            dists.append(h-l)
        return dists


if __name__ == '__main__':
    """
    cd //EfficientAT
    python EfficientAT.py --cuda --model_name=dymn20_as --audio_path=/home/centos/redbeard/test/media-backend/origin_videos/twitter_test_1.wav --head_type=fully_convolutional --resolution 1 --method gaussian --sigma 1.0

    python EfficientAT.py --cuda --model_name=dymn20_as --audio_path=/home/centos/redbeard/test/media-backend/origin_videos/twitter_test_1.wav --head_type=fully_convolutional --resolution 1 --method repeat

    python EfficientAT.py --cuda --model_name=dymn20_as --audio_path=/home/centos/redbeard/test/media-backend/origin_videos/twitter_test_1.wav --head_type=fully_convolutional --resolution 1 --method pad
    """

    import argparse


    parser = argparse.ArgumentParser(description='Example of parser. ')
    # model name decides, which pre-trained model is loaded
    parser.add_argument('--model_name', type=str, default='mn10_as')
    parser.add_argument('--strides', nargs=4, default=[2, 2, 2, 2], type=int)
    parser.add_argument('--head_type', type=str, default="mlp")
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--audio_path', type=Path, required=True)

    # preprocessing
    parser.add_argument('--sample_rate', type=int, default=32000)
    parser.add_argument('--window_size', type=int, default=800)
    parser.add_argument('--hop_size', type=int, default=320)
    parser.add_argument('--n_mels', type=int, default=128)

    # method
    parser.add_argument('--method', type=str, default='gaussian',
                        choices=['gaussian', 'repeat', 'pad'])
    parser.add_argument('--resolution', type=int, default=1)
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--top_rank', type=int, default=10)

    args = parser.parse_args()

    at = EfficientAT(model_name=args.model_name,
                     strides=args.strides,
                     head_type=args.head_type,
                     cuda=args.cuda)
    results = None
    if args.method == 'gaussian':
        results = at.predictGW(args.audio_path,
                               resolution=args.resolution,
                               sigma=args.sigma,
                               top_rank=args.top_rank,
                               n_mels=args.n_mels,
                               sr=args.sample_rate,
                               win_length=args.window_size,
                               hopsize=args.hop_size)
    else:
        results = at.predict(args.audio_path,
                             resolution=args.resolution,
                             pad_method=args.method,
                             top_rank=args.top_rank,
                             n_mels=args.n_mels,
                             sr=args.sample_rate,
                             win_length=args.window_size,
                             hopsize=args.hop_size)

    # Print audio tagging top probabilities
    for i, result in enumerate(list() if results is None else results):
        print(f"****** Acoustic Event Detected: at seconds {i*args.resolution} ******")
        for label, predict in result.items():
            print('{}: {:.3f}'.format(label, predict))
        print("********************************************************")
