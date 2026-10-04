# AI Music Generation using LSTM

## Overview
This project generates new piano music using an LSTM neural network
trained on Chopin piano MIDI files. The model learns note patterns
and predicts new notes one by one to create an original piece.

## Tools Used
- Python
- music21 (MIDI parsing and creation)
- TensorFlow / Keras (LSTM model)
- NumPy, pandas
- pretty_midi and FluidSynth (MIDI to audio)
- Google Colab (GPU training)

## Dataset
MAESTRO dataset (Google Magenta), Chopin piano MIDI files.
15 files used for training.
Source: https://magenta.tensorflow.org/datasets/maestro

## Steps
1. Data collection: downloaded the MAESTRO MIDI-only dataset and
   filtered Chopin files using the metadata CSV.
2. Preprocessing: extracted notes and chords with music21
   (43,754 notes, 1,037 unique notes/chords).
3. Sequence creation: sliding window of 100 notes as input,
   the next note as the target.
4. Model: Embedding layer, 2 LSTM layers (256 units each),
   Dense layers and Dropout (0.3).
5. Training: 30 epochs, batch size 128. Loss dropped from
   5.57 to 2.92.
6. Generation: predicted 200 new notes with temperature 0.8.
7. Output: saved as MIDI and converted to audio (WAV).

## Files in this repository
- AI_music_generation_with_RNNS_and_GNNs_.ipynb : full code (Colab notebook)
- generated_music.mid : generated music (MIDI)
- generated_music.wav : generated music (audio)
- best_model.keras : trained model
- training_loss.png : training output 

## How to run
1. Open the notebook in Google Colab and select a T4 GPU.
2. Upload the MAESTRO MIDI files.
3. Run the cells from top to bottom.

## Limitations
- All notes have the same duration, so the rhythm sounds mechanical.
- Trained on only 15 files.

## Future Improvements
- Add note durations and velocity
- Train on more data for more epochs
- Try Transformer models or GANs (MuseGAN)
