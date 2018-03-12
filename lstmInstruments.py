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
    notes = get_notes_and_instruments()

    # get amount of pitch names
    n_vocab = len(set(notes))

    network_input, network_output = prepare_sequences_with_instruments(notes, n_vocab)

    model = create_network(network_input, n_vocab)

    train(model, network_input, network_output)

def get_notes_and_instruments(): #goal here is to get sequences based on instruments, not just giant sequences of chords
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []

    #beethovenCorpus = corpus.getComposer('beethoven', 'xml')
    #for xml in beethovenCorpus:
    for file in glob.glob("midi_songs/*.mid"):
        #midi = corpus.parse(xml)
        midi = converter.parse(file)
        notes_to_parse = None
        instrumentalNotes = [] #This will hold note objects with instrument info.
        parts = instrument.partitionByInstrument(midi)
        
        if parts: # file has instrument parts. Originally, the assumption is that only one instrument will exist. This is not, in fact, true. So, we will do this:

            # print (len(parts))
            # parts.show('text')
            
            for part in parts:
                instrumentName = part.getInstrument() #BTW, we're assuming that one part will only have one instrument, not multiple embedded into it.
                print (instrumentName)
                notes_to_parse = part.recurse() #THIS IS WHERE WE WILL DIVERGE.
                for elem in notes_to_parse:
                    #for each note, store the instrument that played that note
                    elem.storedInstrument = instrumentName
                    #noteTuple = (instName, elem)
                    instrumentalNotes.append(elem)

                    




        else: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes
            instrumentName = instrument.Piano() #Default instrument shall be a piano.
            for elem in notes_to_parse:
                    #for each note, store the instrument as well.
                    elem.storedInstrument = instrumentName
                    instrumentalNotes.append(elem)



        for element in instrumentalNotes:
            if isinstance(element, note.Note):
                noteTuple = (str(element.pitch), element.storedInstrument.instrumentName) #make tuple that store note pitch and instrument that's playing it.
                notes.append(noteTuple)
                #notes.append(str(element.pitch))

            elif isinstance(element, chord.Chord):
                noteTuple = ('.'.join(str(n) for n in element.normalOrder), element.storedInstrument.instrumentName) #make tuple that stores note pitches and instrument that's playing it.
                notes.append(noteTuple)

    #notes is now a list of tuples (notename, instrument) seen in training.
    with open ("notes", "wb") as filepath:
        pickle.dump(notes, filepath)
            
    
 #    rawPitchInstrumentNames = sorted(notes, key=lambda note: note[0])
    # PitchesAndInstruments = set(rawPitchInstrumentNames) #This is now a set of tuples, each tuple is ("notename", instrument), where notename denotes pitch, and is a string.
    # print (PitchesAndInstruments)
    # with open('PitchesAndInstruments', 'wb') as filepath:
    #   pickle.dump(PitchesAndInstruments, filepath)
    #print (notes)
    return notes

    #return PitchesAndInstruments


def prepare_sequences_with_instruments(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100

    #get all pitch names and instruments. (Basically, get no duplicates.)
    rawPitchInstrumentNames = sorted(notes, key=lambda note: note[0])
    PitchesAndInstruments = set(rawPitchInstrumentNames) #This is now a set of tuples, each tuple is ("notename", instrument), where notename denotes pitch, and is a string.
    

     # create a dictionary to map pitches+instruments to integers
    
    pitchInstrument_to_int = dict ((pitchInstrument, number) for number, pitchInstrument in enumerate(PitchesAndInstruments)) #realize that here, each note is a tuple with the pitch and the instrument object (see PitchesAndInstruments above)

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([pitchInstrument_to_int[char] for char in sequence_in])
        network_output.append(pitchInstrument_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
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