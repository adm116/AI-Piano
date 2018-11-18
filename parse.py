import collections
import os
import glob
import argparse
import pickle
import sys
import numpy
import matplotlib.pyplot as plt
from pathlib import Path
from music21 import converter, instrument, note, chord, stream, interval, pitch
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--dir', help="a directory", type= str)
parser.add_argument('--use_old_notes', help="whether or not to use old pickle notes", type= str, default='false')
parser.add_argument('--limit', help="number of limited files to use", type= int, default=-1)
parser.add_argument('--graph', help="whether to show a plot of frequencies", type= str, default='false')
args = parser.parse_args(sys.argv[1:])

DATA_DIR = args.dir
USE_OLD_NOTES = args.use_old_notes.lower() == 'true'
LIMIT = args.limit
SHOW_GRAPH = args.graph.lower() == 'true'
PICKLE_NOTES = DATA_DIR + '/notes'


def isEighthNoteOffset(element):
    return element.offset % 1 == 0 or element.offset + 0.5 % 1 == 0

def getNotes():
    pickle_notes = Path(PICKLE_NOTES)
    if USE_OLD_NOTES and pickle_notes.is_file():
        print('Using previous notes stored')
        with open(PICKLE_NOTES, 'rb') as filepath:
            return pickle.load(filepath)

    files = []
    for file in glob.glob(DATA_DIR + '/*.mid'):
        files.append(file)

    if LIMIT != -1:
        files = numpy.random.choice(files, LIMIT)

    notes = []
    for file in files:
        midi = converter.parseFile(file)
        key = midi.analyze('key')
        tonic = note.Note(pitch=key.pitchFromDegree(1))
        i = interval.Interval(tonic, note.Note('C'))
        midi = midi.transpose(i)

        notes_to_parse = None
        parts = instrument.partitionByInstrument(midi)
        if parts: # file has instrument parts
        	notes_to_parse = parts.parts[0].recurse()
        else: # file has notes in a flat structure
        	notes_to_parse = midi.flat.notes

        songNotes = []
        offsets = collections.defaultdict(list)
        for element in notes_to_parse:
            if isinstance(element, note.Note) and isEighthNoteOffset(element):
                offsets[element.offset].append(str(element.pitch))
            elif isinstance(element, chord.Chord) and isEighthNoteOffset(element):
                offsets[element.offset].append('.'.join(sorted([str(note.Note(n).pitch) for n in element.normalOrder])))

        for offset in offsets.keys():
            choices = sorted(offsets[offset])
            if len(choices) > 3:
                chosen = numpy.random.choice(choices, 3)
                notes.append('.'.join(str(n) for n in chosen))
            else:
                notes.append('.'.join(str(n) for n in choices))


    with open(PICKLE_NOTES, 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes

def getFreq(notes):
    pitchnames = sorted(set(item for item in notes))
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    freq = collections.defaultdict(int)
    for n in notes:
        freq[note_to_int[n]] += 1
    return freq

def graph(freq):
    x = []
    y = []
    for n in freq.keys():
        x.append(n)
        y.append(freq[n])

    plt.plot(x, y, '.')
    plt.ylabel('number of occurences')
    plt.xlabel('note')
    if SHOW_GRAPH:
        plt.show()

def parse():
    notes = getNotes()
    freq = getFreq(notes)
    graph(freq)

if __name__ == '__main__':
    parse()
