
import os
import sys
import argparse
import tensorflow
import numpy
import pickle
import random
import time
from music21 import converter, instrument, note, chord, stream
from tensorflow.python.keras import utils
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import GRU
from sklearn.utils import shuffle

parser = argparse.ArgumentParser()
parser.add_argument('--dir', help="a directory", type= str)
parser.add_argument('--weights', help="weights file", type= str)
parser.add_argument('--num_notes', help="number of notes to generate", type= int, default=200)
parser.add_argument('--seq_len', help="length of sequences", type= int, default=100)
parser.add_argument('--batch_size', help="batch size", type= int, default=128)
args = parser.parse_args(sys.argv[1:])

DATA_DIR = args.dir
WEIGHTS = 'weights/' + DATA_DIR + '/' + args.weights
NUM_NOTES = args.num_notes
SEQ_LEN = args.seq_len
BATCH = args.batch_size
PICKLE_NOTES = DATA_DIR + '/notes_f'
OUTPUT = 'output/' + DATA_DIR
TOP_NOTES = 25 # choose top TOP_NOTES from predictions

def getProbs(prediction, top):
    probs = []
    total = 0
    for t in top:
        p = prediction[t]
        total += p
        probs.append(p)

    for p in range(0, len(probs)):
        probs[p] = probs[p] / total

    return probs

def generateOutput(network_input, n_vocab, model, pitchnames):
    # Load the weights to each node
    start = numpy.random.randint(0, len(network_input)-1)
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    pattern = network_input[start]

    prediction_output = []
    for note_index in range(NUM_NOTES):
        prediction_input = numpy.reshape(pattern, (1, SEQ_LEN, 1))
        prediction_input = prediction_input / float(n_vocab) # normalize
        prediction = model.predict(prediction_input, batch_size=BATCH, verbose=0)

        # get top TOP_NOTES notes and create prob distribution
        top = numpy.argsort(-prediction[0])[:TOP_NOTES]
        probs = getProbs(prediction[0], top)

        # choose note from top based on probs distribution
        index = numpy.random.choice(top, 1, probs)[0] # numpy.argmax(prediction)
        result = int_to_note[index]
        print(result)
        prediction_output.append(result)
        pattern = numpy.append(pattern, index)
        pattern = pattern[1:]

    return prediction_output

def process(notes, n_vocab, pitchnames):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = SEQ_LEN

     # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    # create input sequences and the corresponding outputs
    network_input = []
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    normalized_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    return (network_input, normalized_input)

def buildNetwork(network_input, n_vocab):
    model = Sequential() # linear stack of layers
    model.add(GRU(128, input_shape=(network_input.shape[1], network_input.shape[2]), activation='softmax', return_sequences=True))
    model.add(GRU(n_vocab, input_shape=(network_input.shape[1], network_input.shape[2]), activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop')
    model.load_weights(WEIGHTS)
    return model

def createMidi(prediction_output):
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        n, offset = pattern
        # pattern is a chord
        if ('.' in n):
            notes_in_chord = n.split('.')
            notes = []

            for current_note in notes_in_chord:
                new_note = note.Note(current_note)
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.volume.velocity = 50
            new_chord.offset = offset
            #new_chord.duration.quarterLength = dur
            output_notes.append(new_chord)

        # pattern is a note
        else:
            new_note = note.Note(n)
            new_note.storedInstrument = instrument.Piano()
            new_note.volume.velocity = 50
            new_note.offset = offset
            #new_note.duration.quarterLength = dur
            output_notes.append(new_note)


    midi_stream = stream.Stream(output_notes)

    if not os.path.exists(OUTPUT):
        os.makedirs(OUTPUT)

    ts = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
    outFile = OUTPUT + '/' + ts + '_output.mid'
    midi_stream.write('midi', fp= outFile)

def generate():
    with open(PICKLE_NOTES, 'rb') as filepath:
        notes = pickle.load(filepath)

    n_vocab = len(set(notes))
    pitchnames = sorted(set(item for item in notes))
    network_input, normalized = process(notes, n_vocab, pitchnames)
    prediction_output = generateOutput(network_input, n_vocab, buildNetwork(normalized, n_vocab), pitchnames)
    createMidi(prediction_output)

if __name__ == '__main__':
    generate()
