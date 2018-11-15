
import os
import sys
import collections
import tensorflow
import numpy
import pickle
from random import randint
from music21 import converter, instrument, note, chord, stream
from tensorflow.python.keras import utils
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import GRU
from tensorflow.python.keras.callbacks import ModelCheckpoint
from pathlib import Path

# specify directory of data, of pickle notes, of weights, num epocks, sequence length
DATA_DIR = sys.argv[1]
PICKLE_NOTES = sys.argv[2]
WEIGHTS_PATH = sys.argv[3]
NEW_PICKLE_NOTES = sys.argv[4]
EPOCHS = int(sys.argv[5])
SEQ_LEN = 100

def getFiles():
    files = []
    for file in os.listdir(DATA_DIR):
    	if file.split(".")[-1] == 'mid':
    		files.append(file)
    return files

def getNotes(files):
    pickle_notes = Path(PICKLE_NOTES)
    if NEW_PICKLE_NOTES == 'false' and pickle_notes.is_file():
        print('Using previous notes')
        with open(PICKLE_NOTES, 'rb') as filepath:
            return pickle.load(filepath)

    notes = []
    for i in range(0, len(files)):
        file = files[i]
        midi = converter.parseFile(DATA_DIR + '/' + file)
        notes_to_parse = None
        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))


    with open(PICKLE_NOTES, 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes

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

    # reshape the input into a format compatible with LSTM layers
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
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

    # 1st param: list of input sequences prepared earlier
    # 2nd param: list of input sequences' respective outputs
    # 3rd param: nb epochs = nb iterations (200 used in tutorial)
    # 4th param: nb of samples in the batch propagated through network
    model.fit(network_input, network_output, epochs=EPOCHS, batch_size=64, callbacks=callbacks_list)

if __name__ == '__main__':
    notes = getNotes(getFiles())
    n_vocab = len(set(notes))
    network_input, network_output = getNetworkInputOuput(notes, n_vocab)
    model = buildNetwork(network_input, n_vocab)
    trainModel(network_input, network_output, model)
