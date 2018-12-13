
import os
import time
import sys
import argparse
import util
import collections
import math
from random import randint
from music21 import converter, instrument, note, chord, stream

START = '#begin'
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

parser = argparse.ArgumentParser()
parser.add_argument('--dir', help="a directory", type= str)
parser.add_argument('--limit', help="number of limited files to use", type= int, default=10)
parser.add_argument('--num_notes', help="number of notes to generate", type= int, default=100)
args = parser.parse_args(sys.argv[1:])

DATA_DIR = args.dir
OUTPUT = 'output/' + DATA_DIR
LIMIT = args.limit
NUM_NOTES = args.num_notes
NOTE_SIZE = 88 # num notes playable on piano


files = []										# all midi files
chosen = set()									# keep track of which files we are using to "train" on
followers = collections.defaultdict(set) 	# what notes follow me ?
counts = collections.defaultdict(int)		# (lastNote, curNote) -> num occurences
bitotalCounts = collections.defaultdict(int) # number of times any note follows a note x

# find all files
for file in os.listdir(DATA_DIR):
	if file.split(".")[-1] == 'mid':
		files.append(file)

for i in range(0, LIMIT):
	file = files[randint(0, len(files) - 1)]
	midi = converter.parseFile(DATA_DIR + '/' + file)
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
		elif isinstance(notes_to_parse[e], chord.Chord):
		    notes.append('.'.join(sorted([str(note.Note(n).pitch) for n in notes_to_parse[e].normalOrder])))

	for e in range(0, len(notes)-1):
		curNote = notes[e]
		nextNote = notes[e+1]
		counts[(curNote, nextNote)] += 1
		bitotalCounts[curNote]+=1
		followers[curNote].add(nextNote)

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
ucs.solve(MusicProblem(costs, NUM_NOTES, followers))
output = ucs.actions

# create output
output_notes = []
offset = 0 # don't stack notes on top of each other
for cur in output:
	if ('.' in cur) or cur.isdigit():
	    notes_in_chord = cur.split('.')
	    notes = []
	    for current_note in notes_in_chord:
	        new_note = note.Note(current_note)
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
	offset += 0.5

midi_stream = stream.Stream(output_notes)
if not os.path.exists(OUTPUT):
	os.makedirs(OUTPUT)

ts = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
outFile = OUTPUT + '/' + ts + '_output.mid'
midi_stream.write('midi', fp= outFile)
