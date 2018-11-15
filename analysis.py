import collections
import os
import pickle
from pathlib import Path
from music21 import converter, instrument, note, chord, stream

info = []

pickle_notes = Path("data/notes_analysis")
if not pickle_notes.is_file():
    print("True")
    with open('data/notes', 'rb') as filepath:
        info = pickle.load(filepath)

else:
    files = []
    for file in os.listdir('beethoven'):
        if file.split(".")[-1] == 'mid':
        	files.append(file)

    # choose limit files at random
    freq = collections.defaultdict(int)
    notes = []
    for file in files:
        midi = converter.parseFile("beethoven/" + file)
        notes_to_parse = None
        parts = instrument.partitionByInstrument(midi)
        if parts: # file has instrument parts
        	notes_to_parse = parts.parts[0].recurse()
        else: # file has notes in a flat structure
        	notes_to_parse = midi.flat.notes

        songNotes = []
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                curNote = str(element.pitch)
                freq[curNote] += 1
                notes.append(curNote)
            elif isinstance(element, chord.Chord):
                curNote = '.'.join(str(n) for n in element.normalOrder)
                freq[curNote] += 1
                notes.append(curNote)

    sequence_length = 10
    seq = collections.defaultdict(int)
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = '.'.join(notes[i:i + sequence_length])
        sequence_out = notes[i + sequence_length]
        seq[(sequence_in, sequence_out)] += 1

    info.append(freq)
    info.append(seq)

    # dump all analysis
    with open('data/notes_analysis', 'wb') as filepath:
        pickle.dump(info, filepath)

freq = info[0]
seq = info[1]
sorted_freq = sorted([(freq[n], n) for n in freq.keys()])
sorted_seq = sorted([(seq[s], s) for s in seq.keys()])
print(sorted_freq, sorted_seq)
