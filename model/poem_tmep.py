
################################################################
# Gru Model Section 
# This section is edited version of Martin Gorner's implementation
# :https://github.com/martin-gorner/tensorflow-rnn-shakespeare
import glob
import sys

# On the gru model the char dictionaries are constant
# so if there are new characters they needs to be added
char_dict = {'â': 7, 'İ': 55, 'î': 26, 'R': 57, 'f': 28, 'a': 5, 'D': 40, 'û': 29, 'n': 3, 'g': 22, 'Î': 64, 'O': 45, 'S': 39, "'": 54, 'ç': 30, '^': 69, 'L': 58, 'd': 9, 'ā': 70, ',': 71, 'e': 1, 'F': 51, 'A': 44, 'v': 25, 'U': 61, 'ş': 23, 'Ş': 48, 'G': 36, 'M': 43, 'c': 27, 'B': 34, 'H': 38, 't': 19, 'I': 62, 'z': 21, 'Â': 50, '.': 63, 's': 12, 'u': 18, 'Y': 46, 'h': 16, 'b': 14, 'r': 4, 'i': 2, '\n': 10, 'p': 32, 'l': 6, '_': 66, 'T': 47, 'P': 59, 'ö': 33, 'E': 42, 'o': 24, 'Ö': 65, 'm': 8, 'ü': 13, 'y': 17, 'ğ': 31, '-': 15, 'ı': 20, 'N': 41, 'Ç': 52, 'K': 37, '’': 35, 'C': 49, ' ': 0, 'ô': 68, 'j': 60, 'V': 56, 'Z': 53, 'Ü': 72, '‘': 67, 'k': 11}

reverse_char_dict = {0: ' ', 1: 'e', 2: 'i', 3: 'n', 4: 'r', 5: 'a', 6: 'l', 7: 'â', 8: 'm', 9: 'd', 10: '\n', 11: 'k', 12: 's', 13: 'ü', 14: 'b', 15: '-', 16: 'h', 17: 'y', 18: 'u', 19: 't', 20: 'ı', 21: 'z', 22: 'g', 23: 'ş', 24: 'o', 25: 'v', 26: 'î', 27: 'c', 28: 'f', 29: 'û', 30: 'ç', 31: 'ğ', 32: 'p', 33: 'ö', 34: 'B', 35: '’', 36: 'G', 37: 'K', 38: 'H', 39: 'S', 40: 'D', 41: 'N', 42: 'E', 43: 'M', 44: 'A', 45: 'O', 46: 'Y', 47: 'T', 48: 'Ş', 49: 'C', 50: 'Â', 51: 'F', 52: 'Ç', 53: 'Z', 54: "'", 55: 'İ', 56: 'V', 57: 'R', 58: 'L', 59: 'P', 60: 'j', 61: 'U', 62: 'I', 63: '.', 64: 'Î', 65: 'Ö', 66: '_', 67: '‘', 68: 'ô', 69: '^', 70: 'ā', 71: ',', 72: 'Ü'}

# the alphabet size
ALPHASIZE = len(char_dict)


# Encodoing and decoding text
def encode_text(s):
    """Encode a string.
    :param s: a text string
    :return: encoded list of code points
    """
    return list(map(lambda a: char_dict[a], s))

def decode_to_text(c):
    """Decode an encoded string.
    :param c: encoded list of code points
    :return:
    """
    return "".join(map(lambda a: reverse_char_dict[a], c))

def sample_from_probabilities(probabilities, topn=ALPHASIZE):
    """Roll the dice to produce a random integer in the [0..ALPHASIZE] range,
    according to the provided probabilities. If topn is specified, only the
    topn highest probabilities are taken into account.
    :param probabilities: a list of size ALPHASIZE with individual probabilities
    :param topn: the number of highest probabilities to consider. Defaults to all of them.
    :return: a random integer
    """
    p = np.squeeze(probabilities)
    p[np.argsort(p)[:-topn]] = 0
    p = p / np.sum(p)
    return np.random.choice(ALPHASIZE, 1, p=p)[0]

def rnn_minibatch_sequencer(raw_data, batch_size, sequence_size, nb_epochs):
    """
    Divides the data into batches of sequences so that all the sequences in one batch
    continue in the next batch. This is a generator that will keep returning batches
    until the input data has been seen nb_epochs times. Sequences are continued even
    between epochs, apart from one, the one corresponding to the end of raw_data.
    The remainder at the end of raw_data that does not fit in an full batch is ignored.
    :param raw_data: the training text
    :param batch_size: the size of a training minibatch
    :param sequence_size: the unroll size of the RNN
    :param nb_epochs: number of epochs to train on
    :return:
        x: one batch of training sequences
        y: on batch of target sequences, i.e. training sequences shifted by 1
        epoch: the current epoch number (starting at 0)
    """
    data = np.array(raw_data)
    data_len = data.shape[0]
    # using (data_len-1) because we must provide for the sequence shifted by 1 too
    nb_batches = (data_len - 1) // (batch_size * sequence_size)
    assert nb_batches > 0, "Not enough data, even for a single batch. Try using a smaller batch_size."
    rounded_data_len = nb_batches * batch_size * sequence_size
    xdata = np.reshape(data[0:rounded_data_len], [batch_size, nb_batches * sequence_size])
    ydata = np.reshape(data[1:rounded_data_len + 1], [batch_size, nb_batches * sequence_size])

    for epoch in range(nb_epochs):
        for batch in range(nb_batches):
            x = xdata[:, batch * sequence_size:(batch + 1) * sequence_size]
            y = ydata[:, batch * sequence_size:(batch + 1) * sequence_size]
            x = np.roll(x, -epoch, axis=0)  # to continue the text from epoch to epoch (do not reset rnn state!)
            y = np.roll(y, -epoch, axis=0)
            yield x, y, epoch


def find_book(index, bookranges):
    return next(
        book["name"] for book in bookranges if (book["start"] <= index < book["end"]))


def find_book_index(index, bookranges):
    return next(
        i for i, book in enumerate(bookranges) if (book["start"] <= index < book["end"]))


def print_learning_learned_comparison(X, Y, losses, bookranges, batch_loss, batch_accuracy, epoch_size, index, epoch):
    """Display utility for printing learning statistics"""
    print()
    # epoch_size in number of batches
    batch_size = X.shape[0]  # batch_size in number of sequences
    sequence_len = X.shape[1]  # sequence_len in number of characters
    start_index_in_epoch = index % (epoch_size * batch_size * sequence_len)
    for k in range(batch_size):
        index_in_epoch = index % (epoch_size * batch_size * sequence_len)
        decx = decode_to_text(X[k])
        decy = decode_to_text(Y[k])
        bookname = find_book(index_in_epoch, bookranges)
        formatted_bookname = "{: <10.40}".format(bookname)  # min 10 and max 40 chars
        epoch_string = "{:4d}".format(index) + " (epoch {}) ".format(epoch)
        loss_string = "loss: {:.5f}".format(losses[k])
        print_string = epoch_string + formatted_bookname + " │ {} │ {} │ {}"
        print(print_string.format(decx, decy, loss_string))
        index += sequence_len
    # box formatting characters:
    # │ \u2502
    # ─ \u2500
    # └ \u2514
    # ┘ \u2518
    # ┴ \u2534
    # ┌ \u250C
    # ┐ \u2510
    format_string = "└{:─^" + str(len(epoch_string)) + "}"
    format_string += "{:─^" + str(len(formatted_bookname)) + "}"
    format_string += "┴{:─^" + str(len(decx) + 2) + "}"
    format_string += "┴{:─^" + str(len(decy) + 2) + "}"
    format_string += "┴{:─^" + str(len(loss_string)) + "}┘"
    footer = format_string.format('INDEX', 'BOOK NAME', 'TRAINING SEQUENCE', 'PREDICTED SEQUENCE', 'LOSS')
    print(footer)
    # print statistics
    batch_index = start_index_in_epoch // (batch_size * sequence_len)
    batch_string = "batch {}/{} in epoch {},".format(batch_index, epoch_size, epoch)
    stats = "{: <28} batch loss: {:.5f}, batch accuracy: {:.5f}".format(batch_string, batch_loss, batch_accuracy)
    print()
    print("TRAINING STATS: {}".format(stats))


class Progress:
    """Text mode progress bar.
    Usage:
            p = Progress(30)
            p.step()
            p.step()
            p.step(start=True) # to restart form 0%
    The progress bar displays a new header at each restart."""
    def __init__(self, maxi, size=100, msg=""):
        """
        :param maxi: the number of steps required to reach 100%
        :param size: the number of characters taken on the screen by the progress bar
        :param msg: the message displayed in the header of the progress bat
        """
        self.maxi = maxi
        self.p = self.__start_progress(maxi)()  # () to get the iterator from the generator
        self.header_printed = False
        self.msg = msg
        self.size = size

    def step(self, reset=False):
        if reset:
            self.__init__(self.maxi, self.size, self.msg)
        if not self.header_printed:
            self.__print_header()
        next(self.p)

    def __print_header(self):
        print()
        format_string = "0%{: ^" + str(self.size - 6) + "}100%"
        print(format_string.format(self.msg))
        self.header_printed = True

    def __start_progress(self, maxi):
        def print_progress():
            # Bresenham's algorithm. Yields the number of dots printed.
            # This will always print 100 dots in max invocations.
            dx = maxi
            dy = self.size
            d = dy - dx
            for x in range(maxi):
                k = 0
                while d >= 0:
                    print('=')
                    sys.stdout.flush()
                    k += 1
                    d -= dx
                d += dy
                yield k

        return print_progress

def create_dictionary(char_list):

    count = collections.Counter(char_list).most_common()
    
    for char, _ in count:
        char_dict[char] = len(char_dict)
    reverse_char_dict = dict(zip(char_dict.values(), char_dict.keys())) 
    print( "char dict: ", char_dict)
    print( "reverse dict :" , reverse_char_dict)
    ALPHASIZE = len(char_dict)


def read_data_files(directory, validation=True):
    """Read data files according to the specified glob pattern
    Optionnaly set aside the last file as validation data.
    No validation data is returned if there are 5 files or less.
    :param directory: for example "data/*.txt"
    :param validation: if True (default), sets the last file aside as validation data
    :return: training data, validation data, list of loaded file names with ranges
     If validation is
    """
    char_list = []
    codetext = []
    bookranges = []
    #shakelist = glob.glob(directory, recursive=True)
    shakelist = glob.glob(directory)
    for shakefile in shakelist:
        shaketext = open(shakefile, "r")
        print("Loading file " + shakefile)
        start = len(codetext)
        s = shaketext.read()
        # update the char list
        #char_list.extend(list(s))
        codetext.extend(encode_text(s))
        end = len(codetext)
        bookranges.append({"start": start, "end": end, "name": shakefile.rsplit("/", 1)[-1]})
        shaketext.close()

    if len(bookranges) == 0:
        sys.exit("No training data has been found. Aborting.")

    #create_dictionary(char_list)

    # For validation, use roughly 90K of text,
    # but no more than 10% of the entire text
    # and no more than 1 book in 5 => no validation at all for 5 files or fewer.

    # 10% of the text is how many files ?
    total_len = len(codetext)
    validation_len = 0
    nb_books1 = 0
    for book in reversed(bookranges):
        validation_len += book["end"]-book["start"]
        nb_books1 += 1
        if validation_len > total_len // 10:
            break

    # 90K of text is how many books ?
    validation_len = 0
    nb_books2 = 0
    for book in reversed(bookranges):
        validation_len += book["end"]-book["start"]
        nb_books2 += 1
        if validation_len > 90*1024:
            break

    # 20% of the books is how many books ?
    nb_books3 = len(bookranges) // 5

    # pick the smallest
    nb_books = min(nb_books1, nb_books2, nb_books3)

    if nb_books == 0 or not validation:
        cutoff = len(codetext)
    else:
        cutoff = bookranges[-nb_books]["start"]
    valitext = codetext[cutoff:]
    codetext = codetext[:cutoff]
    return codetext, valitext, bookranges


def print_data_stats(datalen, valilen, epoch_size):
    datalen_mb = datalen/1024.0/1024.0
    valilen_kb = valilen/1024.0
    print("Training text size is {:.2f}MB with {:.2f}KB set aside for validation.".format(datalen_mb, valilen_kb)
          + " There will be {} batches per epoch".format(epoch_size))


def print_validation_header(validation_start, bookranges):
    bookindex = find_book_index(validation_start, bookranges)
    books = ''
    for i in range(bookindex, len(bookranges)):
        books += bookranges[i]["name"]
        if i < len(bookranges)-1:
            books += ", "
    print("{: <60}".format("Validating on " + books))
    sys.stdout.flush()


def print_validation_stats(loss, accuracy):
    print("VALIDATION STATS:                                  loss: {:.5f},       accuracy: {:.5f}".format(loss,
                                                                                                           accuracy))


def print_text_generation_header():
    print()
    print("┌{:─^111}┐".format('Generating random text from learned state'))


def print_text_generation_footer():
    print()
    print("└{:─^111}┘".format('End of generation'))


def frequency_limiter(n, multiple=1, modulo=0):
    def limit(i):
        return i % (multiple * n) == modulo*multiple
    return limit
