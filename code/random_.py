
import os
import sys
import pickle
import collections
import argparse
import time
from random import randint
from music21 import converter, instrument, note, chord, stream

parser = argparse.ArgumentParser()
parser.add_argument('--dir', help="a directory", type= str)
args = parser.parse_args(sys.argv[1:])

DATA_DIR = args.dir
PICKLE_NOTES = DATA_DIR + '/notes'
OUTPUT = 'output/' + DATA_DIR

notes = []
with open(PICKLE_NOTES, 'rb') as filepath:
	notes = pickle.load(filepath)

notes = set(notes)
notesList = list(notes)
offset = 0
output_notes = []

for i in range(0, 100):
	cur = notesList[randint(0, len(notesList) - 1)]
	print(cur)
	if ('.' in cur) or cur.isdigit():
		notes_in_chord = cur.split('.')
		notes = []
		for current_note in notes_in_chord:
		    new_note = note.Note(current_note)
		    new_note.storedInstrument = instrument.Piano()
		    notes.append(new_note)
		new_chord = chord.Chord(notes)
		new_chord.offset = offset
		new_chord.volume.velocity = 50
		output_notes.append(new_chord)
	else:
		new_note = note.Note(cur)
		new_note.offset = offset
		new_note.volume.velocity = 50
		new_note.storedInstrument = instrument.Piano()
		output_notes.append(new_note)

	offset += 0.5

midi_stream = stream.Stream(output_notes)
if not os.path.exists(OUTPUT):
	os.makedirs(OUTPUT)

ts = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
outFile = OUTPUT + '/' + ts + '_output.mid'
midi_stream.write('midi', fp= outFile)
