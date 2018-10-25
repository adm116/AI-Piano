
import os
import collections
from random import randint
from music21 import converter, instrument, note, chord, stream

notes = set()
files = []
limit = 10

# find all files
for file in os.listdir("data"):
	if file.split(".")[-1] == 'mid':
		files.append(file)

# choose limit files at random
for i in range(0, limit):
	file = files[i]
	midi = converter.parseFile("data/" + file)
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
			notes.add(curNote)
		elif isinstance(element, chord.Chord):
			curNote = '.'.join(str(n) for n in element.normalOrder)
			notes.add(curNote)

# choose all random notes
offset = 0 # don't stack notes on top of each other
output_notes = []
notesList = list(notes)
for i in range(0, 50):
    cur = notesList[randint(0, len(notesList))]
    if ('.' in cur) or cur.isdigit():
	    notes_in_chord = cur.split('.')
	    notes = []
	    for current_note in notes_in_chord:
	        new_note = note.Note(int(current_note))
	        new_note.storedInstrument = instrument.Piano()
	        notes.append(new_note)
	    new_chord = chord.Chord(notes)
	    new_chord.offset = offset
	    output_notes.append(new_chord)
    else:
        new_note = note.Note(cur)
        new_note.offset = offset
        new_note.storedInstrument = instrument.Piano()
        output_notes.append(new_note)

    cur = notesList[randint(0, len(notesList))]
    offset += 0.5

midi_stream = stream.Stream(output_notes)
midi_stream.write('midi', fp='test_output.mid')
