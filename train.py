
import os
import collections
import tensorflow
import numpy
import pickle
from random import randint
from music21 import converter, instrument, note, chord, stream
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Activation
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ModelCheckpoint

LIMIT = 16 # limit number of files read in for now
DATA_DIR = 'beethoven'
EPOCHS = 15

def getFiles():
    files = []
    for file in os.listdir(DATA_DIR):
    	if file.split(".")[-1] == 'mid':
    		files.append(file)
    return files

def getNotes(files):
    notes = []
    for i in range(0, LIMIT):
    	file = files[i]
    	midi = converter.parseFile(DATA_DIR + '/' + file)
    	notes_to_parse = None
    	parts = instrument.partitionByInstrument(midi)
    	if parts: # file has instrument parts
    		notes_to_parse = parts.parts[0].recurse()
    	else: # file has notes in a flat structure
    		notes_to_parse = midi.flat.notes

    	for element in notes_to_parse:
            # single note
    		if isinstance(element, note.Note):
    			curNote = str(element.pitch)
    			notes.append(curNote)
            # chord
    		elif isinstance(element, chord.Chord):
    			curNote = '.'.join(str(n) for n in element.normalOrder)
    			notes.append(curNote)

    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes

def getNetworkInputOuput(notes):
    sequence_length = 100
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
    # reshape the input into a format compatible with LSTM layers
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    network_input = network_input / float(len(notes))
    network_output = utils.to_categorical(network_output)
    return (network_input, network_output)


# Modeling
def buildNetwork(network_input, n_vocab):
    """
    for the 1st layer (input tensor) we need to specify its shape (must have the same shape as training data)
    shape is inferred for future layers.... THIS IS WHAT WE CAN TUNE (add/remove layers, tune params)
    """

    model = Sequential() # linear stack of layers

    # Current model has: 3 LSTM layers, 3 Dropout layers, 2 Dense layers, 1 activation layer
    model.add(LSTM(
        512, # number of nodes in each later (NEED TO CHANGE)
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    )) # returns a sequence of vectors of dimension 512
    model.add(Dropout(0.3)) # fraction of input units that should be dropped during training
    model.add(LSTM(512, return_sequences=True)) # returns a sequence of vectors of dimension 512
    model.add(Dropout(0.3))
    model.add(LSTM(512)) # returns a single vector of dimension 512
    model.add(Dense(256)) # Dense(x) is a fully-connected layer with x hidden units
    model.add(Dropout(0.3))
    model.add(Dense(151)) # (nb nodes last layer) = (nb of outputs in system) --> output of the network will map to pitches.
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model

def trainModel(network_input, network_output, n_vocab, model):
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"

    # use checkpoints to save weights of the network nodes after every epoch
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    # 1st param: list of input sequences prepared earlier
    # 2nd param: list of input sequences' respective outputs
    # 3rd param: nb epochs = nb iterations (200 used in tutorial)
    # 4th param: nb of samples in the batch propagated through network
    model.fit(network_input, network_output, epochs=EPOCHS, batch_size=64, callbacks=callbacks_list)

if __name__ == '__main__':
    notes = getNotes(getFiles())
    network_input, network_output = getNetworkInputOuput(notes)
    model = buildNetwork(network_input, len(notes))
    trainModel(network_input, network_output, len(notes), model)
