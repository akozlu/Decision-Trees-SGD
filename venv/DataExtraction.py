import numpy as np


def loadCorpus(path):
    # function to laod the data
    name_label_dict = {}
    with open(path) as myfile:

        for line in myfile:

            line1 = line.rstrip('\n')
            data = line1.split(" ", 1)
            if data[0] == "+":

                name_label_dict[data[1]] = int(1)
            else:

                name_label_dict[data[1]] = int(0)

    return [name_label_dict, list(name_label_dict.keys())]


def load_test_data(path):
    with open(path) as f:
        lst = [line.rstrip() for line in f]

    f.close()
    return [dict(), lst]


# This class transform names into boolean vectors (as np arrays) and associate it with labels
# For each name, ten feature types are created.
# These  feature  types  are  Boolean  in  nature,  and  are  indicators
# for  the  first  five characters from the first and last names.
# Suppose  youwant to extract features corresponding to the first letter “n” in the first name,you  will
# have  26  Boolean  features,  one  for  each  letter  in  the  alphabe
class DataExtraction(object):

    def __init__(self, data, namelist):

        # name-label dictionary
        self.name_label_dict = data

        self.test_name_list = namelist
        self.test_name_vector_dict = {}
        # get the namelist to iterate through the names in names_to_vector function
        self.name_list = namelist

        # the vector-label dictionary will be stored here
        self.vector_label_dictionary = {}

        self.X = np.array([], dtype=int)  # self.X will store data as (700,260) vector
        self.Y = np.array([], dtype=int)  # self.Y will store labels as (700,) vector

    def get_name_list(self):

        return (self.name_list)  # returns the name list

    def get_test_name_list(self):

        return self.test_name_list

    def get_label(self, name):

        return self.name_label_dict[name]

    def get_vector_label_dict(self):

        return self.vector_label_dictionary

    def get_test_name_vector_dict(self):
        return self.test_name_vector_dict

    def get_X(self):
        return self.X

    def get_Y(self):

        return self.Y

        ## unused method

    def is_vowel(self, x):

        if (x in ('a', 'e', 'i', 'o', 'u')):
            return 1
        else:
            return 0

    def count_vowels(self, name):

        vowel = ['a', 'e', 'i', 'o', 'u']
        count = 0
        for letter in name:
            if letter in vowel:
                count += 1

        return count

    def test_names_to_vector(self):

        for name in self.get_test_name_list():

            # print(name)
            # Function returns names into vectors of length 260
            original = name
            # for every name get last and first
            name = name.split()
            first_name = name[0]
            last_name = name[1]

            # initial np arrays
            final_first_name_array = np.array([], dtype=int)

            final_last_name_array = np.array([], dtype=int)

            zero_vector = np.zeros((26,), dtype=int)

            if len(first_name) >= 5:

                first_name_characters = [character for character in first_name[:5]]

                # for every character in first name
                for ch in first_name_characters:
                    # get their corresponding vector of length 26
                    char_vector = self.vector_for_Character(ch)

                    # add it to the final array that will have size 130 and will represent first 5 characters of first name
                    final_first_name_array = np.concatenate((final_first_name_array, char_vector))


            else:

                n = len(first_name)
                first_name_characters = [character for character in first_name]

                for ch in first_name_characters:
                    char_vector = self.vector_for_Character(ch)

                    final_first_name_array = np.concatenate((final_first_name_array, char_vector))

                # append 26 zeros (zero_vector) for the remaining letters
                for i in range(0, 5 - n):
                    final_first_name_array = np.concatenate((final_first_name_array, zero_vector))

            # do the same for the last name

            if len(last_name) >= 5:

                last_name_characters = [character for character in last_name[:5]]

                for ch in last_name_characters:
                    char_vector = self.vector_for_Character(ch)
                    final_last_name_array = np.concatenate((final_last_name_array, char_vector))

            else:
                n = len(last_name)

                last_name_characters = [character for character in last_name]

                for ch in last_name_characters:
                    char_vector = self.vector_for_Character(ch)

                    final_last_name_array = np.concatenate((final_last_name_array, char_vector))

                for i in range(0, 5 - n):
                    final_last_name_array = np.concatenate((final_last_name_array, zero_vector))

            # append first and last name arrays together to get the final array
            final_array = np.concatenate((final_first_name_array, final_last_name_array))

            new_arr = (tuple((final_array)))

            self.test_name_vector_dict[(new_arr)] = original  # Data,label pairs will be stored here.

    def name_to_Vector(self):

        # Function returns names into vectors of length 260

        for name in self.get_name_list():

            original = name
            # for every name get last and first
            name = name.split()
            first_name = name[0]
            last_name = name[1]

            # This part was used to extract extra features and test certain conjunction of features to improve accuracy
            length_of_the_name = len(first_name) + len(last_name)

            additional_features = list()

            # is_vowel = self.is_vowel(last_name[:1])
            # additional_features.append(is_vowel)
            # is_vowel = self.is_vowel(first_name[:1])
            # additional_features.append(is_vowel)

            additional_features.append(length_of_the_name)

            # initial np arrays
            final_first_name_array = np.array([], dtype=int)

            final_last_name_array = np.array([], dtype=int)

            zero_vector = np.zeros((26,), dtype=int)

            if len(first_name) >= 5:

                first_name_characters = [character for character in first_name[:5]]

                # for every character in first name
                for ch in first_name_characters:
                    # get their corresponding vector of length 26
                    char_vector = self.vector_for_Character(ch)

                    # add it to the final array that will have size 130 and will represent first 5 characters of first name
                    final_first_name_array = np.concatenate((final_first_name_array, char_vector))

                number_of_vowels = self.count_vowels(first_name)
            else:

                number_of_vowels = self.count_vowels(first_name)

                n = len(first_name)
                first_name_characters = [character for character in first_name]

                for ch in first_name_characters:
                    char_vector = self.vector_for_Character(ch)

                    final_first_name_array = np.concatenate((final_first_name_array, char_vector))

                # append 26 zeros (zero_vector) for the remaining letters
                for i in range(0, 5 - n):
                    final_first_name_array = np.concatenate((final_first_name_array, zero_vector))

            # do the same for the last name

            if len(last_name) >= 5:

                last_name_characters = [character for character in last_name[:5]]

                for ch in last_name_characters:
                    char_vector = self.vector_for_Character(ch)
                    final_last_name_array = np.concatenate((final_last_name_array, char_vector))

                number_of_vowels = number_of_vowels + self.count_vowels(last_name)
            else:
                n = len(last_name)

                number_of_vowels = number_of_vowels + self.count_vowels(last_name)

                last_name_characters = [character for character in last_name]

                for ch in last_name_characters:
                    char_vector = self.vector_for_Character(ch)

                    final_last_name_array = np.concatenate((final_last_name_array, char_vector))

                for i in range(0, 5 - n):
                    final_last_name_array = np.concatenate((final_last_name_array, zero_vector))

            # append first and last name arrays together to get the final array
            final_array = np.concatenate((final_first_name_array, final_last_name_array))

            # add additional features
            # additional_features.append(number_of_vowels)
            # final_array = np.concatenate((final_array, np.asarray(additional_features)))

            new_arr = ((tuple(final_array)))

            label = self.get_label(original)  # get label of the name
            self.vector_label_dictionary[new_arr] = label  # Data,label pairs will be stored here.

    def vector_for_Character(self, character):

        # helper function to determine the position of the letter in the alphabet and return the corresponding numpy array
        # for that letter.

        # define an alphabet
        alfa = "abcdefghijklmnopqrstuvwxyz"

        # initially we have 26 zeros
        char_vector = np.zeros((26,), dtype=int)
        # define reverse lookup dict

        rdict = dict([(x[1], x[0]) for x in enumerate(alfa)])
        character_position = rdict[character]

        # we change the boolean value in the position of the character
        np.put(char_vector, character_position, 1)
        return (char_vector)

    def get_X_Y(self):

        self.name_to_Vector()
        dict = self.get_vector_label_dict()
        X = []
        Y = []
        for key, value in dict.items():
            X.append(list(key))
            Y.append(value)
        X = np.array(X)
        Y = np.array(Y)
        self.X = X
        self.Y = Y
