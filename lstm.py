
import os
import collections
import tensorflow
import numpy
from tensorflow.keras import utils
from random import randint
from music21 import converter, instrument, note, chord, stream
import pickle
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

LIMIT = 2 # limit number of files read in

def getFiles():
    files = []
    for file in os.listdir("data"):
    	if file.split(".")[-1] == 'mid':
    		files.append(file)
    return files

def getNotes(files):
    notes = []
    for i in range(0, LIMIT):
    	file = files[i]
    	midi = converter.parseFile("data/" + file)
    	notes_to_parse = None
    	parts = instrument.partitionByInstrument(midi)
    	if parts: # file has instrument parts
    		notes_to_parse = parts.parts[0].recurse()
    	else: # file has notes in a flat structure
    		notes_to_parse = midi.flat.notes

    	for element in notes_to_parse:
    		if isinstance(element, note.Note):
    			curNote = str(element.pitch)
    			notes.append(curNote)
    		elif isinstance(element, chord.Chord):
    			curNote = '.'.join(str(n) for n in element.normalOrder)
    			notes.append(curNote)
    return notes

def getNetworkInputOuput(notes):
    seq_len = 100
    pitchnames = sorted(set(item for item in notes))
    note_to_int = dict((note, num) for num, note in enumerate(pitchnames))
    network_input = []
    network_output = []

    for i in range(0, len(notes) - seq_len):
        seq_in = notes[i:i+seq_len]
        seq_out = notes[i+seq_len]
        network_input.append([note_to_int[char] for char in seq_in])
        network_output.append(note_to_int[seq_out])

    n_patterns = len(network_input)
    network_input = numpy.reshape(network_input, (n_patterns, seq_len, 1))
    network_input = network_input / float(len(notes))
    network_output = utils.to_categorical(network_output)
    return (network_input, network_output)

def trainNetwork():
    files = getFiles()
    notes = getNotes(files)
    n_vocab = len(set(notes)) # nb of pitches
    network_input, network_output = getNetworkInputOuput(notes)
    print(network_input, network_output)
    model = buildNetwork(network_input, n_vocab)
    trainModel(model, network_input, network_output)


# choose all random notes
#offset = 0 # don't stack notes on top of each other
#output_notes = []

#midi_stream = stream.Stream(output_notes)
#midi_stream.write('midi', fp='test_output.mid')


# Modeling
def buildNetwork(network_input, n_vocab):
    model = Sequential() # linear stack of layers

    """ 
    for the 1st layer (input tensor) we need to specify its shape (must have the same shape as training data)
    shape is inferred for future layers 
    """

    # THIS IS WHAT WE CAN TUNE (add/remove layers, tune params)

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
    model.add(Dense(n_vocab)) # (nb nodes last layer) = (nb of outputs in system) --> output of the network will map to pitches.
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop') 

    return model

def trainModel(model, network_input, network_output):
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
    model.fit(network_input, network_output, epochs=15, batch_size=64, callbacks=callbacks_list)

if __name__ == '__main__':
    trainNetwork()
