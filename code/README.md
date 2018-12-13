Parsing Notes:
  In order to run anything, you must first run parse.py which will parse the
MIDI files of a folder into notes/chords and store them in a pickle files to
be read in by any of the models. The command for this is:

    python3 parse.py --dir data

  There are also optional commands for limiting the number of files to be read in
for graphing the distributions of notes, and whether we are using an existing pickle
file. An example is:

    python3 parse.py --dir data --use_old_notes true --limit 100 --graph true

##############################################################################

Generate Random Notes:
  In order to generate random notes as a baseline, the following command can
be run:

    python3 random_.py --dir data

##############################################################################

Generate notes with search problem:
  Our second baseline is a search problem. To run that, use the command:

    python3 search.py --dir data

  This requires the util.py file in order to run UCS. Additional arguments include
the number of notes to generate and the limit of files to be read in:

    python3 search.py --dir data --limit 100 --num_notes 100

##############################################################################

GRU Model:
  To run our ultimate model, we need to first train our model and then generate
music. To do so, first run:

    python3 train.py --dir data

  Additional arguments include specifying the number of epochs, the batch size,
the sequence length to feed into the GRU, and the number of nodes in the first GRU
layer:

    python3 train.py --dir data  --epochs 100 --batch_size 128 --seq_len 25 --hidden 128

  In order to generate music based off of this model, run the command where the weights
specified are the weights file generated from training:

    python3 predict.py --dir data --weights weights.hdf5

  Additional arguments include specifying number of notes to generate, sequence length,
batch size, number of hidden nodes in the first GRU layer, and the number of top notes
to sample from the prediction vector (all arguments used in training should match those
used here). For example:

    python3 predict.py --dir data --weights weights.hdf5 --num_notes 100 --seq_len 25
      --batch_size 128 --hidden 128 --top_notes 25
