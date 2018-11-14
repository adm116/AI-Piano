
import os
import tensorflow
import numpy
import pickle
from music21 import converter, instrument, note, chord, stream
from train import buildNetwork
from train import getNetworkInputOuput

NOTES = 200 # num notes generated

def generateOutput(network_input, n_vocab, model):
    # Load the weights to each node
    model.load_weights('weights/weights-improvement-01-4.6388-bigger.hdf5')

    pitchnames = sorted(set(item for item in notes))
    start = numpy.random.randint(0, len(network_input)-1)
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    pattern = network_input[start]

    print(network_input[0])
    prediction_output = []
    for note_index in range(NOTES):
        print(pattern)
        prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
        #prediction_input = prediction_input / float(n_vocab)
        prediction = model.predict(prediction_input, verbose=0)
        index = numpy.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)
        pattern = numpy.append(pattern, index)
        pattern = pattern[1:len(pattern)]

    return prediction_output

def createMidi(prediction_output):
    output_notes = []
    offset = 0
    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='test_output.mid')

if __name__ == '__main__':
    with open('data/notes', 'rb') as filepath:
        notes = pickle.load(filepath)

    n_vocab = len(set(notes))
    network_input, network_output = getNetworkInputOuput(notes, n_vocab)
    prediction_output = generateOutput(network_input, n_vocab, buildNetwork(network_input, n_vocab))
    createMidi(prediction_output)