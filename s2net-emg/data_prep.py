import pandas
import os

dir_name = os.path.dirname(__file__)
file_train = "1_raw_data_1.txt"
file_test = "2_raw_data_1.txt"
# open TRAIN file
file_df = pandas.read_csv('unsplit_data/' + file_train, delimiter='\t', header=None,
                          names=['t', 'ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6',
                                'ch7', 'ch8', 'class'], skiprows=[0])
print(file_df[:10])

train_lst_dir = os.path.join(dir_name, "data/train/training_list.txt")
train_lst = open(train_lst_dir, 'a')

# open TEST file
file_df_test = pandas.read_csv('unsplit_data/' + file_test, delimiter='\t', header=None,
                          names=['t', 'ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6',
                                'ch7', 'ch8', 'class'], skiprows=[0])
print(file_df_test[:10])

test_lst_dir = os.path.join(dir_name, "data/train/testing_list.txt")
test_lst = open(test_lst_dir, 'a')

df_len = 150
for i in range(1, 9):
    print("Processing class " + str(i))
    filt_df = file_df.loc[file_df['class'] == i]
    filt_df_test = file_df_test.loc[file_df_test['class'] == i]

    train_df_dir = os.path.join(dir_name, "data/train/")
    os.mkdir(train_df_dir + str(i))
    test_df_dir = os.path.join(dir_name, "data/test/")
    os.mkdir(test_df_dir + str(i))
    # split TRAIN set
    nmb = 0
    for j in range(0, (filt_df.shape[0] // df_len) * df_len, df_len):
        cut_df = filt_df.iloc[j:j + df_len]

        fl_name = "train_class_" + str(i) + '_' + str(nmb) + ".csv"
        fl_loc = "data/train/" + str(i) + "/" + fl_name

        train_lst.writelines(fl_name + "\n")

        cut_df.to_csv(fl_loc, index=False)
        nmb += 1
    # split TEST set
    nmb = 0
    for j in range(0, (filt_df_test.shape[0] // df_len) * df_len, df_len):
        cut_df = filt_df_test.iloc[j:j + df_len]

        fl_name = "test_class_" + str(i) + '_' + str(nmb) + ".csv"
        fl_loc = "data/test/" + str(i) + "/" + fl_name

        test_lst.writelines(fl_name + "\n")

        cut_df.to_csv(fl_loc, index=False)
        nmb += 1

train_lst.close()
test_lst.close()

print("Processing STREAM file")
# open STREAM file
stream_dir = os.path.join(dir_name, "unsplit_data/" + file_test)
file_stream = open(stream_dir, 'r')
to_dir = os.path.join(dir_name, "raw/" + file_test)
to_file = open(to_dir, "a")
line = file_stream.readline()
line = file_stream.readline()
while line != "":
    if line[-2] != "0" and line[-2] != "9":
        to_file.writelines(line)
    line = file_stream.readline()
file_stream.close()
to_file.close()

print("Process finished")
