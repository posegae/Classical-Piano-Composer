""" This module generates notes for a midi file using the
    trained neural network """
import pickle
import numpy
from music21 import instrument, note, stream, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation

def generate():
    """ Generate a piano midi file """
    #load the notes used to train the model
    with open('notes', 'rb') as filepath:
        notes = pickle.load(filepath)


    
    # Get all pitch names and corresponding instruments
    rawPitchInstrumentNames = sorted(notes, key=lambda note: note[0])
    PitchesAndInstruments = set(rawPitchInstrumentNames) #This is now a set of tuples, each tuple is ("notename", instrument), where notename denotes pitch, and is a string.

    n_vocab = len(set(notes))

    network_input, normalized_input = prepare_sequences_with_instruments(notes, PitchesAndInstruments, n_vocab)
    model = create_network(normalized_input, n_vocab)
    prediction_output = generate_notes_with_instruments(model, network_input, PitchesAndInstruments, n_vocab)
    create_midi_with_instruments(prediction_output)

def prepare_sequences_with_instruments(notes, PitchesAndInstruments, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    # map between notes and integers and back
    pitchInstrument_to_int = dict ((pitchInstrument, number) for number, pitchInstrument in enumerate(PitchesAndInstruments)) #realize that here, each note is a tuple with the pitch and the instrument object (see PitchesAndInstruments above)

    sequence_length = 100
    network_input = []
    output = []
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([pitchInstrument_to_int[char] for char in sequence_in])
        output.append(pitchInstrument_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    normalized_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    normalized_input = normalized_input / float(n_vocab)

    return (network_input, normalized_input)

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

    # Load the weights to each node
    model.load_weights('best-weights-without-rests.hdf5')

    return model

def generate_notes_with_instruments(model, network_input, PitchesAndInstruments, n_vocab):
    """ Generate notes from the neural network based on a sequence of notes """
    # pick a random sequence from the input as a starting point for the prediction
    start = numpy.random.randint(0, len(network_input)-1)

    int_to_note = dict((number, note) for number, note in enumerate(PitchesAndInstruments))

    pattern = network_input[start]
    prediction_output = []

    # generate 500 notes
    for note_index in range(500):
        prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)

        index = numpy.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return prediction_output

def create_midi_with_instruments(prediction_output):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    offset = 0
    output_notes = []
    midi_stream = stream.Stream()

    #realize now that prediction output is a series of tuples of the form (notename, instrumentname)

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        
        #pattern[0] is note or chord, pattern[1] is the instrument name (stored as a string)

        # pattern is a chord
        if ('.' in pattern[0]) or pattern[0].isdigit():
            notes_in_chord = pattern[0].split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                if pattern[1] != None:
                    try:
                        new_note.storedInstrument = instrument.fromString(pattern[1])
                        midi_stream.append(instrument.fromString(pattern[1]))
                    except Exception as e:
                        print (e)
                        midi_stream.append(instrument.Ukulele())
                    print ("Non Piano Instrument found!")
                else: 
                    #we will default the instrument to accordion.
                    
                    midi_stream.append(instrument.Accordion())
                    
                new_note.show('text')
                
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            #output_notes.append(new_chord)
            midi_stream.append(new_chord)
        # pattern is a note
        else:
            new_note = note.Note(pattern[0])
            new_note.offset = offset
            if pattern[1] != None:
                print (pattern[1])
                try:
                     new_note.storedInstrument = instrument.fromString(pattern[1])
                     midi_stream.append(instrument.Ukulele())
                except Exception as e:
                    print (e)
                    midi_stream.append(instrument.Ukulele())
                print ("There's a non-piano instrument in here!")
            else: 
            #here our default will be piano
                midi_stream.append(instrument.Piano())
        
            midi_stream.append(new_note)
           # output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += 0.5

    #midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp='test_output.mid')

if __name__ == '__main__':
    generate()
