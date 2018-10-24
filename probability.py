
import os
import collections
from random import randint
from music21 import converter, instrument, note, chord, stream

notes = []
files = []
limit = 10
chosen = set()
followers = collections.defaultdict(set) 	# what notes follow me ?
counts = collections.defaultdict(int)		# (lastNote, curNote) -> num occurences

# find all files
for file in os.listdir("data"):
	if file.split(".")[-1] == 'mid':
		files.append(file)

# choose limit files at random and build up followers map
for i in range(0, limit):
	filesIndex = 0
	while True:
		filesIndex = randint(0, len(files)-1)
		if filesIndex not in chosen:
			break
	chosen.add(filesIndex)
	file = files[filesIndex]

	midi = converter.parseFile("data/" + file)
	notes_to_parse = None
	parts = instrument.partitionByInstrument(midi)
	if parts: # file has instrument parts
		notes_to_parse = parts.parts[0].recurse()
	else: # file has notes in a flat structure
		notes_to_parse = midi.flat.notes
	
	lastNote = None
	for element in notes_to_parse:
		if isinstance(element, note.Note):
			curNote = str(element.pitch)
			if lastNote != None:
				counts[(lastNote, curNote)] += 1
				followers[lastNote].add(curNote)
			lastNote = curNote
			notes.append(curNote)

# choose a starting point at random
# for every other note, choose a random note out of those that come after the previous note
offset = 0 # don't stack notes on top of each other
output_notes = []
cur = notes[randint(0, len(notes))]
for i in range(0, 50):
	new_note = note.Note(cur)
	new_note.offset = offset
	new_note.storedInstrument = instrument.Piano()
	output_notes.append(new_note)

	# 80% chance we just take the best note to follow, otherwise choose a random note from the note's followers
	if randint(0, 1) == 1:
		cur = max( [(counts[(cur, neighbor)], neighbor) for neighbor in followers[cur]] )[1]
	else:
		possibleNextNotes = [(counts[(cur, neighbor)], neighbor) for neighbor in followers[cur]]
		cur = possibleNextNotes[randint(0, len(possibleNextNotes)-1)][1]
	offset += 0.5

midi_stream = stream.Stream(output_notes)
midi_stream.write('midi', fp='test_output.mid')