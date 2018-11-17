import collections
import os
import glob
import sys
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from music21 import converter, instrument, note, chord, stream
from pathlib import Path

DATA_DIR = sys.argv[1]  # data directory
PICKLE_NOTES = DATA_DIR + '/notes'                     # note file to put pickle info
USE_PICKLE_NOTES = sys.argv[2] == 'true'        # false if don't want to read from old pickle file else true

def getNotes():
    pickle_notes = Path(PICKLE_NOTES)
    if USE_PICKLE_NOTES and pickle_notes.is_file():
        print('Using previous notes stored')
        with open(PICKLE_NOTES, 'rb') as filepath:
            return pickle.load(filepath)

    notes = []
    for file in glob.glob(DATA_DIR + '/*.mid'):
        midi = converter.parseFile(file)
        notes_to_parse = None
        parts = instrument.partitionByInstrument(midi)
        if parts: # file has instrument parts
        	notes_to_parse = parts.parts[0].recurse()
        else: # file has notes in a flat structure
        	notes_to_parse = midi.flat.notes

        songNotes = []
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    with open(PICKLE_NOTES, 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes


notes = getNotes()
pitchnames = sorted(set(item for item in notes))
note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
freq = collections.defaultdict(int)
for n in notes:
    freq[note_to_int[n]] += 1

x = []
y = []
for n in freq.keys():
    x.append(n)
    y.append(freq[n])

plt.plot(x, y, '.')
plt.ylabel('number of occurences')
plt.xlabel('note')
plt.show()
