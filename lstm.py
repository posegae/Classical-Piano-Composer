""" This module prepares midi file data and feeds it to the neural
    network for training """
from pprint import pprint
import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

def train_network():
    """ Train a Neural Network to generate music """
    notes = get_notes()

    # get amount of pitch names
    n_vocab = len(set(notes))

    network_input, network_output = prepare_sequences(notes, n_vocab)

    model = create_network(network_input, n_vocab)

    train(model, network_input, network_output)

def get_notes():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory
        NEW: notes is now a list of 2-ples containing pitch(es) and offset"""
    notes = []

    for file in glob.glob("midi_music_pop/*.mid"):
        midi = converter.parse(file)

        notes_to_parse = None

        parts = instrument.partitionByInstrument(midi)

        if parts: # file has instrument parts
            notes_to_parse = parts.parts[0].recurse()
        else: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                pitch = str(element.pitch)
                # CHANGES: start accounting for offset
                offset = str(element.offset)
                notes.append((pitch, offset))
            elif isinstance(element, chord.Chord):
                pitches = '.'.join(str(n) for n in element.normalOrder)
                offset = str(element.offset)
                notes.append((pitches, offset))

    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes

def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100

    # get all pitch names
    pitchnames = sorted(set(item[0] for item in notes))

     # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input_notes = []
    network_input_offsets = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length][0]
        # sequence_out = [note[0] for note in sequence_out]

        notes_ints = [note_to_int[char[0]] for char in sequence_in]
        offsets = [char[1] for char in sequence_in]

        network_input_notes.append(notes_ints)
        network_input_offsets.append(offsets)
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input_notes)

    # reshape the input into a format compatible with LSTM layers
    network_input = numpy.reshape(network_input_notes, (n_patterns, sequence_length, 1))

    # pprint(network_input)

    # normalize input
    network_input = network_input / float(n_vocab)

    network_output = np_utils.to_categorical(network_output)

    return (network_input, network_output)

def create_network(network_input, n_vocab):
    """ create the structure of the neural network """

    print(network_input.shape)

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

    model.fit(network_input, network_output, epochs=200, batch_size=64, callbacks=callbacks_list, verbose=1)

if __name__ == '__main__':
    train_network()
