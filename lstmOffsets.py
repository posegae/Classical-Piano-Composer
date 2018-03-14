""" This module prepares midi file data and feeds it to the neural
    network for training """
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
    notes = get_notes_and_offsets()

    # get amount of pitch names
    n_vocab = len(set([tuple(n) for n in notes]))

    network_input, network_output = prepare_sequences_with_offsets(notes, n_vocab)

    model = create_network(network_input, n_vocab)

    train(model, network_input, network_output)
def get_notes_and_offsets():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory
        Also gets the offsets

        Here, offset really refers to the length of time between each note.
        """
    notes = []

    for file in glob.glob('midi_music_pop/*.mid'):
        midi = converter.parse(file)
        notes_to_parse = None
        parts = instrument.partitionByInstrument(midi)
        if parts:
            notes_to_parse = parts.parts[0].recurse()
        else:
            notes_to_parse = midi.flat.notes

        prev_note_offset = 0
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                pitch = str(element.pitch)
                offset = element.offset - prev_note_offset
                notes.append([pitch, offset])
                prev_note_offset += offset
            elif isinstance(element, chord.Chord):
                pitches = '.'.join(str(n) for n in element.normalOrder)
                offset = element.offset - prev_note_offset
                notes.append([pitches, offset])
                prev_note_offset += offset

    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    # print(notes)

    return notes

def prepare_sequences_with_offsets(notes, n_vocab):
    ''' Prepare the sequences used by the NN. Adapted to account for offsets '''
    sequence_length = 100
    raw_pitch_offset_names = sorted([tuple(n) for n in notes], key=lambda x: x[0])
    raw_pitch_offset_names = set(raw_pitch_offset_names)

    pitches = [n[0] for n in notes]
    pitches = sorted(set(pitches))
    offsets = [n[1] for n in notes]
    offsets = sorted(set(offsets))


    # dict that maps pitches to numbers
    pitch_to_int = dict((pitches, number) for number, pitches in enumerate(pitches))

    # dict that maps offset distances to numbers
    offset_to_int = dict((offsets, number) for number, offsets in enumerate(offsets))

    network_input = []
    network_output = []

    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([[pitch_to_int[char[0]], offset_to_int[char[1]]] for char in sequence_in])
        network_output.append([pitch_to_int[sequence_out[0]], offset_to_int[sequence_out[1]]])
    n_patterns = len(network_input)

    # the 2 on the end specifies that we have 2 dimensions, or features to look at.
    #  in this particular file, they would be pitches and offset differences
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 2))

    network_input = network_input / float(n_vocab)

    # print(network_input.shape)
    # network_output = np_utils.to_categorical(network_output)

    # print(network_output)

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
