""" This module prepares midi file data and feeds it to the neural
    network for training """
import os
import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord, corpus
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

def train_network():
    """ Train a Neural Network to generate music """
    notes = get_notes_and_durations()

    # get amount of pitch names
    n_vocab = len(set(notes))

    network_input, network_output = prepare_sequences_with_durations(notes, n_vocab)

    model = create_network(network_input, n_vocab)

    train(model, network_input, network_output)
def get_notes_and_durations():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory
        Also gets the durations"""
    notes = []

    for file in glob.glob('midi_music_pop/*.mid'):
        midi = converter.parse(file)
        notes_to_parse = None
        parts = instrument.partitionByInstrument(midi)
        if parts:
            notes_to_parse = parts.parts[0].recurse()
        else:
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                pitch = str(element.pitch)
                duration = str(float(element.duration.quarterLength))
                notes.append((pitch, duration))
            elif isinstance(element, chord.Chord):
                pitches = '.'.join(str(n) for n in element.normalOrder)
                duration = str(float(element.duration.quarterLength))
                notes.append((pitches, duration))

    try:
        os.makedirs('data')
    except:
        pass
    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes

def prepare_sequences_with_durations(notes, n_vocab):
    ''' Prepare the sequences used by the NN. Adapted to account for durations '''
    sequence_length = 100
    raw_pitch_duration_names = sorted(notes, key=lambda x: x[0])
    raw_pitch_duration_names = set(notes)

    # Dictionary that maps (pitch, duration) tuples to integers
    pitch_duration_to_int = dict((raw_pitch_duration_names, number) for number, raw_pitch_duration_names in enumerate(raw_pitch_duration_names))

    network_input = []
    network_output = []

    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([pitch_duration_to_int[char] for char in sequence_in])
        network_output.append(pitch_duration_to_int[sequence_out])
    n_patterns = len(network_input)

    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))

    network_input = network_input / float(n_vocab)

    network_output = np_utils.to_categorical(network_output)
    return (network_input, network_output)


def create_network(network_input, n_vocab):
    """ create the structure of the neural network """
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model

def train(model, network_input, network_output):
    """ train the neural network """
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, epochs=200, batch_size=64, callbacks=callbacks_list)

if __name__ == '__main__':
    train_network()
