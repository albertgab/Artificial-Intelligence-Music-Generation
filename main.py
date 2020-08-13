import os

from keras.callbacks import ModelCheckpoint
from keras.engine.saving import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras import layers
from keras import models

from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from subprocess import call

list_of_songs = []
trn_set = []
multiplier = 0
seq_len = 128


def read_from_file(file_title):
    os.chdir('D:\\OneDrive - Aberystwyth University\\Projects\\Major Project\\training data')
    os.system("midicsv " + file_title + ".mid " + file_title + ".txt")
    f = open(file_title + ".txt", "r")
    csv_file = f.read()
    f.close()
    os.remove(file_title + ".txt")
    return csv_file


""""
def file_compression(tmp_file):
    index = 0
    # replacing 'Note_on_c' by 1
    while index != -1:
        prev_index = index + 15
        index = tmp_file[prev_index:].find('Note_on_c')
        tmp_file = tmp_file[:index + prev_index] + '1' + tmp_file[index + 9 + prev_index:]
    index = 0
    # replacing 'Note_off_c' by 0
    while index != -1:
        prev_index = index + 15
        index = tmp_file[prev_index:].find('Note_off_c')
        tmp_file = tmp_file[:index + prev_index] + '0' + tmp_file[index + 10 + prev_index:]
    index = 0
    # removing ', 127'
    while index != -1:
        prev_index = index + 15
        index = tmp_file[prev_index:].find(', 127')
        tmp_file = tmp_file[:index + prev_index] + tmp_file[index + 5 + prev_index:]

    # saving to txt file
    f = open(file_title + ".txt", "a")
    f.truncate(0)
    f.write(tmp_file)
    f.close()

    return tmp_file

"""


def find_start_end(src_str):
    """
    # reading one line at a time
    line = 0
    ch = tmp_str[0]
    mel_start = 0   # character which starts line with first note
    for i in range(len(tmp_str)):
        if ch == '\n':
            line += 1
            mel_start = ch
        if ch == 'N':
            j = 0
            # checking if following letters are 'Note_on_c'
            for c in 'Note_on_c':
                if tmp_str[i+j] != c:
                    break
                j += 1


        ch = tmp_str[i]

    lines = [line][1]
    i = 0
    line = 0
    line_arr = 0
    while i <= len(tmp_str):

        while ch != '\n' or i <= len(tmp_str):


            ch = tmp_str[i]
            i += 1
    line += 1
    """
    i = src_str.find("Note_on_c")
    mel_start = src_str.rfind("\n", 0, i) + 1
    i = src_str.find("End_track", i)
    mel_end = src_str.rfind("\n", 0, i)

    return mel_start, mel_end


def extract_track(start, end, src_str):
    number_lines = src_str.count("\n", start, end) + 1
    arr_track = [[0] * 4 for i in range(number_lines)]

    i = 0
    j = start
    val = 0
    for i in range(number_lines):

        #        if arr_track[1][0] == 0 and i>1:
        #            print(0)

        j = src_str.find(",", j) + 2
        ch = src_str[j]
        while ch != ",":
            j += 1
            val = int(ch) + val * 10
            ch = src_str[j]

        #       if val == 0 and i >1:
        #            print(0)

        arr_track[i][0] = val
        val = 0
        j += 2

        if "Note_on_c" not in src_str[j:j + 9] and "Note_off_c" not in src_str[j:j + 10]:

            print(src_str[j:j + 10])

            # Sometimes there are lines that are inside track, but have to be avoid like pitch bends
            j = src_str.find("\n", j)
            j += 2
        else:

            # print(src_str[j:j + 10])

            if "Note_on_c" in src_str[j:j + 9]:
                arr_track[i][1] = 1
            elif "Note_off_c" in src_str[j:j + 10]:
                arr_track[i][1] = 0

            j = src_str.find(",", j) + 2
            j = src_str.find(",", j) + 2  # intentional repetition
            ch = src_str[j]
            while ch != ",":
                j += 1
                val = int(ch) + val * 10
                ch = src_str[j]
            arr_track[i][2] = val
            val = 0

            j += 2
            ch = src_str[j]
            while ch != "\n":
                j += 1
                val = int(ch) + val * 10
                ch = src_str[j]
            arr_track[i][3] = val
            val = 0

            j += 1

    return arr_track


def extract_melody(arr_track):
    melody = []
    start_note = [arr_track[0][0], arr_track[0][2]]  # time in millis, pitch
    i = 1
    for i in range(len(arr_track)):
        # ending note, same pitch as already started note
        if arr_track[i][1] == 0 and arr_track[i][2] == start_note[1]:
            melody.append([arr_track[i][0] - start_note[0], start_note[1]])
            start_note = [arr_track[i][0], 0]
        # starting note, no note already started
        elif arr_track[i][1] == 1 and start_note[1] == 0:
            if arr_track[i][0] - start_note[0] != 0:
                melody.append([arr_track[i][0] - start_note[0], 0])
            start_note = [arr_track[i][0], arr_track[i][2]]
        # starting note, note is higher than already started
        elif arr_track[i][1] == 1 and arr_track[i][2] > start_note[1]:
            melody.append([arr_track[i][0] - start_note[0], start_note[1]])
            start_note = [arr_track[i][0], arr_track[i][2]]
        # some files set volume as 0 which means end of note,
        # so if the pitch is the same end volume is 0 it treats it as end of note
        elif arr_track[i][1] == 1 and arr_track[i][2] == start_note[1] and arr_track[i][3] == 0:
            melody.append([arr_track[i][0] - start_note[0], start_note[1]])
            start_note = [arr_track[i][0], 0]
    return melody


def train_network(trn_set):
    trn_set = trn_set[:int(len(trn_set) / seq_len) * seq_len]
    trn_set = np.reshape(trn_set, (int(len(trn_set) / seq_len), seq_len, 88))
    out_set = np.zeros((trn_set.shape[0], 88))
    for i in range(0, len(trn_set) - 1):
        out_set[i] = trn_set[i + 1][0]
    print(out_set.shape)
    model = Sequential()
    model.add(LSTM(256, input_shape=(trn_set.shape[1], trn_set.shape[2]), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(88))
    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])

    os.chdir('D:\\OneDrive - Aberystwyth University\\Projects\\Major Project\\nets')
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"

    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    model.summary()
    model.fit(trn_set, out_set, epochs=200, batch_size=64, callbacks=callbacks_list)


"""
    # evaluate the keras model
    _, accuracy = model.evaluate(trn_set, y)
    print('Accuracy: %.2f' % (accuracy * 100))
    classes = model.predict(y, batch_size=28)
    print(classes)
"""


def generating_music(trn_set):
    #reused code from traning
    trn_set = trn_set[:int(len(trn_set) / seq_len) * seq_len]
    trn_set = np.reshape(trn_set, (int(len(trn_set) / seq_len), seq_len, 88))
    out_set = np.zeros((trn_set.shape[0], 88))
    for i in range(0, len(trn_set) - 1):
        out_set[i] = trn_set[i + 1][0]
    model = Sequential()
    model.add(LSTM(256, input_shape=(trn_set.shape[1], trn_set.shape[2]), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(88))
    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])

    os.chdir('D:\\OneDrive - Aberystwyth University\\Projects\\Major Project\\nets')
    model = load_model('whole_data-188-0.0000-bigger.hdf5')
    sum_start = 0

    print(trn_set.shape)
    # takes 1 or 2 notes from randomly chosen point from training data

    sequence = trn_set[np.random.randint(0, len(trn_set) - 1)]
    print(sequence.shape)
    gen_track = np.zeros((512, 88), dtype=np.int8)
    sequence = sequence.reshape(1, 128, 88)
    # generating of 512 frames - 32 bars
    for frame_ind in range(512):
        prediction = model.predict(sequence, verbose=0)
        print(prediction)
        note = np.where(prediction == np.amax(prediction))
        note = note[1]
        print(note)
        #print(note.shape)
        #print(note[1])
        gen_track[frame_ind][note] = 1
        for i in range(128):
            if i == 127:
                sequence[0][i] = note
            else:
                sequence[0][i] = sequence[0][i + 1]



    print(gen_track)
    array_to_midi(gen_track)







def array_to_midi(arr):
    cont_notes = []
    str_to_save = "0, 0, Header, 1, 2, 240\n" \
                  "1, 0, Start_track\n" \
                  "1, 0, Time_signature, 4, 2, 24, 8\n" \
                  "1, 0, Tempo, 625000\n" \
                  "1, 1, End_track\n" \
                  "2, 0, Start_track\n"

    for i in range(len(arr)):
        for j in range(88):
            if arr[i][j] == 1:
                if j not in cont_notes:
                    str_to_save += "2, " + str(i * 96) + ", Note_on_c, 0, " + str(j) + ", 64\n"
                    cont_notes.append(j)

        for note in cont_notes:
            if arr[i][note] == 0:
                str_to_save += "2, " + str(i * 96) + ", Note_off_c, 0, " + str(note) + ", 64\n"
                while note in cont_notes:
                    cont_notes.remove(note)

    str_to_save += "2, " + str(len(arr) * 96) + ", End_track\n0, 0, End_of_file\n"

    f = open("generated_music.txt", "a")
    f.truncate(0)
    f.write(str_to_save)
    f.close()
    os.system("csvmidi generated_music.txt generated_music.mid")

# finally this method seemed to not work very well in this case, so I decided to not use them
def net_single_input():
    global trn_set
    os.chdir('D:\\OneDrive - Aberystwyth University\\Projects\\Major Project\\training data')
    files = os.listdir()
    for file_title in files:
        file_title = file_title.split(".")
        file_title = file_title[0]
        src_str = read_from_file(file_title)
        rest = 0
        while src_str[rest:].find("Note_on_c") != -1:
            start, end = find_start_end(src_str[rest:])
            arr_track = extract_track(start, end, src_str[rest:])

            rest = rest + end
            # print(len(arr_track))
            # print(arr_track)
            trn_set.extend(extract_melody(arr_track))
            trn_set.extend([[1440, 0]])
    trn_set = np.array(trn_set)
    timestamp = 1
    trn_set = np.reshape(trn_set, (trn_set.shape[0], 1, trn_set.shape[1]))
    # print(trn_set)
    # print(len(trn_set))
    # print(trn_set.shape)
    train_network(trn_set)


def shape_trn_data(trn_track, arr_track):
    cont_notes = []
    for i in range(len(arr_track) - 1):

        # print(arr_track[i][0] / multiplier)

        if arr_track[i][1] == 1 and arr_track[i][3] != 0:
            trn_track[round(arr_track[i][0] / multiplier), arr_track[i][2]] = 1
            cont_notes.append(arr_track[i][2])
        elif (arr_track[i][1] == 0 and arr_track[i][2] in cont_notes) or \
                (arr_track[i][1] == 1 and arr_track[i][3] == 0):
            done = False
            j = round(arr_track[i][0] / multiplier) - 1
            while not done:
                #               if trn_track[j][arr_track[i][2]] == 1:
                while arr_track[i][2] in cont_notes:
                    cont_notes.remove(arr_track[i][2])
                done = True
                """
                else:

                    if j==0 and i >1:
                        print(0)

                    trn_track[j][arr_track[i][2]] = 1
                    j -= 1
                    """
    return trn_track


def net_multi_input():
    global trn_set, trn_track, multiplier
    os.chdir('D:\\OneDrive - Aberystwyth University\\Projects\\Major Project\\training data')
    files = os.listdir()
    for file_title in files:
        file_title = file_title.split(".")
        file_title = file_title[0]
        src_str = read_from_file(file_title)
        rest = 0
        first_pass = True
        j = 0
        for i in range(5):
            j += src_str[j:].find(",") + 1
        j += 1
        val = 0
        while src_str[j] != "\n":
            val = int(src_str[j]) + val * 10
            j += 1
        multiplier = int(val / 4)
        print(multiplier)
        val = 0

        while src_str[rest:].find("Note_on_c") != -1:
            start, end = find_start_end(src_str[rest:])
            arr_track = extract_track(start, end, src_str[rest:])
            if first_pass:
                trn_track = np.array(
                    np.zeros((int(arr_track[len(arr_track) - 1][0] / multiplier) + 1, 88), dtype=np.int8))
                first_pass = False
            else:
                if len(trn_track) < int(arr_track[len(arr_track) - 1][0] / multiplier):
                    # Change shape and size of array in-place, without reallocating
                    trn_track.resize((int(arr_track[len(arr_track) - 1][0] / multiplier) + 1, 88))
                    # print(np.resize(trn_track(int(arr_track[len(arr_track) - 1][0] / multiplier) + 1, 88)))

            trn_track = shape_trn_data(trn_track, arr_track)
            rest = rest + end

        list_of_songs.extend(trn_track)
        list_of_songs.extend(np.zeros((16, 88), dtype=np.int8))

        # print(shape(list_of_songs))
        # list_of_songs1 = np.array(list_of_songs)
        # print(list_of_songs1.shape)

    # trn_set.extend(np.zeros((16, 88), dtype=np.int8))
    trn_set = np.array(list_of_songs, dtype=np.int8)
    print(trn_set.shape)
    print(trn_set)


def main():
    # net_single_input()
    net_multi_input()
    #train_network(trn_set)
    generating_music(trn_set)
    # array_to_midi(trn_set)


if __name__ == "__main__":
    main()
