
import os
import util
import collections
import math
from random import randint
from music21 import converter, instrument, note, chord, stream
START = '#begin'
LIMIT = 10
NOTE_SIZE = 88 # num notes playable on piano

##############################################################################
#Model
class MusicProblem(util.SearchProblem):
	def __init__(self, costs, totalNotes, nextNotes):
		self.costs = costs
		self.totalNotes = totalNotes
		self.nextNotes = nextNotes

	def startState(self):
		return ((START, 0))

	def isEnd(self, state):
		return state[1] == self.totalNotes

	def succAndCost(self, state):
		lastNote, totalSoFar = state
		return [(next, (next, totalSoFar + 1), costs[(lastNote, next)]) for next in self.nextNotes[lastNote]]

##############################################################################
# Setting up problem

files = []										# all midi files
chosen = set()									# keep track of which files we are using to "train" on
followers = collections.defaultdict(set) 	# what notes follow me ?
counts = collections.defaultdict(int)		# (lastNote, curNote) -> num occurences
bitotalCounts = collections.defaultdict(int) # number of times any note follows a note x

# find all files
for file in os.listdir("data"):
	if file.split(".")[-1] == 'mid':
		files.append(file)

for i in range(0, LIMIT):
	# choose file at random, but only if not chosen already
	filesIndex = 0
	while True:
		filesIndex = randint(0, len(files)-1)
		if filesIndex not in chosen:
			break
	chosen.add(filesIndex)
	file = files[filesIndex]

	# parse it
	midi = converter.parseFile("data/" + file)
	notes_to_parse = None
	parts = instrument.partitionByInstrument(midi)
	if parts:
		notes_to_parse = parts.parts[0].recurse()
	else:
		notes_to_parse = midi.flat.notes

	notes = []
	for e in range(0, len(notes_to_parse)):
		if isinstance(notes_to_parse[e], note.Note):
			notes.append(str(notes_to_parse[e].pitch))
	for e in range(0, len(notes)-1):
		curNote = notes[e]
		nextNote = notes[e+1]
		counts[(curNote, nextNote)] += 1
		bitotalCounts[curNote]+=1
		followers[nextNote].add(curNote)

		# if beginning note
		if e == 0:
			bitotalCounts[START]+=1
			counts[(START, curNote)] += 1
			followers[START].add(curNote)

# build costs
costs = collections.defaultdict(int)
for cur in followers:
	for next in followers[cur]:
		costs[(cur, next)] = math.log(bitotalCounts[cur] + NOTE_SIZE) - math.log(counts[(cur, next)] + 1)

##############################################################################
# solve problem
ucs = util.UniformCostSearch(verbose=1)
ucs.solve(MusicProblem(costs, 50, followers))
output = ucs.actions
print(output)

# create output
output_notes = []
offset = 0 # don't stack notes on top of each other
for cur in output:
	new_note = note.Note(cur)
	new_note.offset = offset
	new_note.storedInstrument = instrument.Piano()
	output_notes.append(new_note)
	offset += 0.5

midi_stream = stream.Stream(output_notes)
midi_stream.write('midi', fp='test_output.mid')
