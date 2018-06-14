import numpy as np


class Dataset:

    def __init__(self):
        pass

    def preprocess(self, input_file):
        with open(input_file, "r") as f:
            data = f.read().decode("utf-8") # python 2

        # count and sort most frequent characters

        # self.sorted chars contains just the characters ordered descending by frequency
        self.char2id = dict(zip(self.sorted_chars, range(len(self.sorted_chars)))) 
        # reverse the mapping
        self.id2char = {k:v for v,k in self.char2id.items()}
        # convert the data to ids
        self.x = np.array(list(map(self.char2id.get, data)))


    def encode(self, sequence):
        # returns the sequence encoded as integers
        pass


    def decode(self, encoded_sequence):
        # returns the sequence decoded as letters
        pass


    def create_minibatches(self):
        self.num_batches = int(len(self.x) / (self.batch_size * self.sequence_length)) # calculate the number of batches

        # Is all the data going to be present in the batches? Why?
        # What happens if we select a batch size and sequence length larger than the length of the data?

        #######################################
        #       Convert data to batches       #
        #######################################

        pass


    def next_minibatch(self):
        # ...

        batch_x, batch_y = None, None
        # handling batch pointer & reset
        # new_epoch is a boolean indicating if the batch pointer was reset
        # in this function call
        return new_epoch, batch_x, batch_y


