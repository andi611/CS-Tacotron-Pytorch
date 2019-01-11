import argparse
import os
from multiprocessing import cpu_count
from tqdm import tqdm
from hparams import hparams
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
from utils import audio


def preprocess(args):
  meta_path = args.meta_path
  wav_dir = args.wav_dir
  out_dir = args.out_dir
  os.makedirs(out_dir, exist_ok=True)
  metadata = build_from_path(meta_path, wav_dir, out_dir, args.num_workers, tqdm=tqdm)
  write_metadata(metadata, out_dir)


def write_metadata(metadata, out_dir):
  with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
    for m in metadata:
      f.write('|'.join([str(x) for x in m]) + '\n')
  frames = sum([m[2] for m in metadata])
  hours = frames * hparams.frame_shift_ms / (3600 * 1000)
  print('Wrote %d utterances, %d frames (%.2f hours)' % (len(metadata), frames, hours))
  print('Max input length:  %d' % max(len(m[3]) for m in metadata))
  print('Max output length: %d' % max(m[2] for m in metadata))


def build_from_path(meta_path, wav_dir, out_dir, num_workers=1, tqdm=lambda x: x):
  '''Preprocesses the LJ Speech dataset from a given input path into a given output directory.

    Args:
      in_dir: The directory where you have downloaded the LJ Speech dataset
      out_dir: The directory to write the output into
      num_workers: Optional number of worker processes to parallelize across
      tqdm: You can optionally pass tqdm to get a nice progress bar

    Returns:
      A list of tuples describing the training examples. This should be written to train.txt
  '''

  # We use ProcessPoolExecutor to parallize across processes. This is just an optimization and you
  # can omit it and just call _process_utterance on each input if you want.
  executor = ProcessPoolExecutor(max_workers=num_workers)
  futures = []
  index = 1
  with open(meta_path, encoding='utf-8') as f:
    for line in f:
      parts = line.strip().split('|')
      wav_path = os.path.join(wav_dir, '%s.wav' % parts[0])
      text = parts[2]
      futures.append(executor.submit(partial(_process_utterance, out_dir, index, wav_path, text)))
      index += 1
  return [future.result() for future in tqdm(futures)]


def _process_utterance(out_dir, index, wav_path, text):
  '''Preprocesses a single utterance audio/text pair.

  This writes the mel and linear scale spectrograms to disk and returns a tuple to write
  to the train.txt file.

  Args:
    out_dir: The directory to write the spectrograms into
    index: The numeric index to use in the spectrogram filenames.
    wav_path: Path to the audio file containing the speech input
    text: The text spoken in the input audio file

  Returns:
    A (spectrogram_filename, mel_filename, n_frames, text) tuple to write to train.txt
  '''

  # Load the audio to a numpy array:
  wav = audio.load_wav(wav_path)

  # Compute the linear-scale spectrogram from the wav:
  spectrogram = audio.spectrogram(wav).astype(np.float32)
  n_frames = spectrogram.shape[1]

  # Compute a mel-scale spectrogram from the wav:
  mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)

  # Write the spectrograms to disk:
  spectrogram_filename = 'ljspeech-spec-%05d.npy' % index
  mel_filename = 'ljspeech-mel-%05d.npy' % index
  np.save(os.path.join(out_dir, spectrogram_filename), spectrogram.T, allow_pickle=False)
  np.save(os.path.join(out_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)

  # Return a tuple describing this training example:
  return (spectrogram_filename, mel_filename, n_frames, text)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--meta_path', type=str, required=True)
  parser.add_argument('--wav_dir', type=str, required=True)
  parser.add_argument('--out_dir', type=str, default='training')
  parser.add_argument('--num_workers', type=int, default=cpu_count())
  args = parser.parse_args()
  preprocess(args)


if __name__ == "__main__":
  main()
