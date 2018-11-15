
import os
import collections
import tensorflow
import numpy
import pickle
from pathlib import Path
from random import randint
from music21 import converter, instrument, note, chord, stream
from tensorflow.python.keras import utils
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import GRU
from tensorflow.python.keras.callbacks import ModelCheckpoint

LIMIT = 16 # limit number of files read in for now
DATA_DIR = 'beethoven'
EPOCHS = 20
NOTES_PER_TIMESTEP = 3
SEQ_LEN = 20

def getFiles():
    files = []
    for file in os.listdir(DATA_DIR):
    	if file.split(".")[-1] == 'mid':
    		files.append(file)
    return files

def getNotes(files):
    timesteps = collections.defaultdict(list)
    last = 0
    for i in range(0, LIMIT):
        file = files[i]
        midi = converter.parseFile(DATA_DIR + '/' + file)
        notes_to_parse = None
        parts = instrument.partitionByInstrument(midi)
        if parts: # file has instrument parts
        	notes_to_parse = parts.parts[0].recurse()
        else: # file has notes in a flat structure
        	notes_to_parse = midi.flat.notes

        lastNote = None
        for element in notes_to_parse:
            # single note
            if isinstance(element, note.Note):
                cur_note = str(element.pitch)
                lastNote = element
                timesteps[last + element.offset].append(cur_note)
            # chord
            elif isinstance(element, chord.Chord):
                lastNote = element
                for n in element.normalOrder:
                    timesteps[last + element.offset].append(str(n))
        last = last + lastNote.offset + 0.5

    notes = []
    for time in timesteps.keys():
        cur_notes = timesteps[time]
        if len(cur_notes) > NOTES_PER_TIMESTEP:
            # choose NOTES_PER_TIMESTEP random notes
            chosen = sorted(numpy.random.choice(cur_notes, NOTES_PER_TIMESTEP))
            notes.append('.'.join(chosen))
        else:
            chosen = sorted(cur_notes)
            notes.append('.'.join(n for n in chosen))

    with open(DATA_DIR + '/notes_timesteps', 'wb') as filepath:
        pickle.dump(notes, filepath)

    print(notes)
    return notes

def getNetworkInputOuput(notes, n_vocab):
    # get all pitch names
    pitchnames = sorted(set(item for item in notes))
    # create a dictionary to map pitches to integers
    note_to_int = dict((item, number) for number, item in enumerate(pitchnames))

    network_input = []
    network_output = []
    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - SEQ_LEN, 1):
        sequence_in = notes[i:i + SEQ_LEN]
        sequence_out = notes[i + SEQ_LEN]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)
    # reshape the input into a format compatible with LSTM layers
    network_input = numpy.reshape(network_input, (n_patterns, SEQ_LEN, 1))
    # normalize input
    #network_input = network_input / float(n_vocab)
    network_output = utils.to_categorical(network_output)
    return (network_input, network_output)


# Modeling
def buildNetwork(network_input, n_vocab):
    """
    for the 1st layer (input tensor) we need to specify its shape (must have the same shape as training data)
    shape is inferred for future layers.... THIS IS WHAT WE CAN TUNE (add/remove layers, tune params)
    """
    model = Sequential() # linear stack of layers
    model.add(GRU(n_vocab, input_shape=(network_input.shape[1], network_input.shape[2]), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model

def trainModel(network_input, network_output, model):
    filepath = "weights/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"

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
