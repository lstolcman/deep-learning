import numpy as np


class Dataset:

    def __init__(self, minibatch_size, sequence_length):
        self.batch_size = minibatch_size
        self.sequence_length = sequence_length
        self.batch_index = 0

    def preprocess(self, input_file):
        with open(input_file, "r") as f:
            data = f.read()##.decode("utf-8") # python 2

        # count and sort most frequent characters
        ## https://stackoverflow.com/questions/20510768/count-frequency-of-words-in-a-list-and-sort-by-frequency
        from collections import Counter
        counter = Counter(data)
        ##https://stackoverflow.com/questions/20950650/how-to-sort-counter-by-value-python
        '''
        Outside of counters, sorting can always be adjusted based on a key function;
        .sort() and sorted() both take callable that lets you specify a value on which
        to sort the input sequence; sorted(x, key=x.get, reverse=True) would give you
        the same sorting as x.most_common(), but only return the keys, for example:
        >>> sorted(x, key=x.get, reverse=True)
        ['c', 'a', 'b']
        '''
        self.sorted_chars = sorted(counter, key=counter.get, reverse=True)

        # self.sorted chars contains just the characters ordered descending by frequency
        self.char2id = dict(zip(self.sorted_chars, range(len(self.sorted_chars)))) 
        # reverse the mapping
        self.id2char = {k:v for v,k in self.char2id.items()}
        # convert the data to ids
        self.x = np.array(list(map(self.char2id.get, data)))


    def encode(self, sequence):
        # returns the sequence encoded as integers
        return [self.char2id[char] for char in sequence]
        #pass


    def decode(self, encoded_sequence):
        # returns the sequence decoded as letters
        return [self.id2char[_id] for _id in encoded_sequence]
        #pass


    def create_minibatches(self):
        self.num_batches = int(len(self.x) / (self.batch_size * self.sequence_length)) # calculate the number of batches
        self.batches = []

        batch_step = self.batch_size*self.sequence_length
        for i in range(self.num_batches):
            batch_start = batch_step * i
            batch_end = batch_start + batch_step
            '''
            In the task of language modelling, our target is
            simply the input sequence, but shifted by one symbol
            Therefore, for the input x_t the target y_t we are trying to predict is actually x_t+1.
            '''
            _input = np.array(self.x[batch_start:batch_end])
            target = np.array(self.x[batch_start+1:batch_end+1])
            self.batches.append((_input, target))

        self.batch_index = 0
        # Is all the data going to be present in the batches? Why?
        # What happens if we select a batch size and sequence length larger than the length of the data?

        #######################################
        #       Convert data to batches       #
        #######################################


    def next_minibatch(self):
        new_epoch = self.batch_index == self.num_batches
        if new_epoch:
            self.batch_index = 0

        batch_x, batch_y = self.batches[self.batch_index]
        self.batch_index += 1

        #batch_x, batch_y = None, None
        # handling batch pointer & reset
        # new_epoch is a boolean indicating if the batch pointer was reset
        # in this function call
        return new_epoch, batch_x, batch_y

