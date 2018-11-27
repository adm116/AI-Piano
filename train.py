
import os
import sys
import glob
import collections
import tensorflow
import numpy
import pickle
import argparse
from random import randint
from music21 import converter, instrument, note, chord, stream
from tensorflow.python.keras import utils
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import GRU
from tensorflow.python.keras.callbacks import ModelCheckpoint
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--dir', help="a directory", type= str)
parser.add_argument('--epochs', help="number of epochs", type= int, default=200)
parser.add_argument('--seq_len', help="length of sequences", type= int, default=100)
parser.add_argument('--batch_size', help="batch size", type= int, default=128)
args = parser.parse_args(sys.argv[1:])

DATA_DIR = args.dir
EPOCHS = args.epochs
SEQ_LEN = args.seq_len
BATCH = args.batch_size
PICKLE_NOTES = DATA_DIR + '/notes'          # note file to put pickle info
WEIGHTS_PATH = 'weights/' + DATA_DIR        # path for where to put weights

def getNetworkInputOuput(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = SEQ_LEN

    # get all pitch names
    pitchnames = sorted(set(item for item in notes))

     # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    network_input = network_input / float(n_vocab)
    network_output = utils.to_categorical(network_output)
    return (network_input, network_output)

# Modeling
def buildNetwork(network_input, n_vocab):
    model = Sequential() # linear stack of layers
    model.add(GRU(n_vocab, input_shape=(network_input.shape[1], network_input.shape[2]), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model

def trainModel(network_input, network_output, model):
    if not os.path.exists(WEIGHTS_PATH):
        os.makedirs(WEIGHTS_PATH)

    filepath = WEIGHTS_PATH + '/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5'

    # use checkpoints to save weights of the network nodes after every epoch
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]
    model.fit(network_input, network_output, epochs=EPOCHS, batch_size=BATCH, callbacks=callbacks_list)

def train():
    with open(PICKLE_NOTES, 'rb') as filepath:
        notes = pickle.load(filepath)

    n_vocab = len(set(notes))
    network_input, network_output = getNetworkInputOuput(notes, n_vocab)
    model = buildNetwork(network_input, n_vocab)
    print("Training model...")
    trainModel(network_input, network_output, model)

if __name__ == '__main__':
    train()
