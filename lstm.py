
import os
import collections
import tensorflow
import numpy
from tensorflow.keras import utils
from random import randint
from music21 import converter, instrument, note, chord, stream

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


files = getFiles()
notes = getNotes(files)
network_input, network_output = getNetworkInputOuput(notes)
print(network_input, network_output)


# choose all random notes
#offset = 0 # don't stack notes on top of each other
#output_notes = []

#midi_stream = stream.Stream(output_notes)
#midi_stream.write('midi', fp='test_output.mid')
