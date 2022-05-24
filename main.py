import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn import preprocessing

# reading train data from file
# and transforming it into the desired format

input_file = open("data/train_samples.txt", "rt", encoding="utf-8")
training_input = input_file.read().splitlines()
for i in range(len(training_input)):
    training_input[i] = training_input[i].split("\t")[1]
training_input = np.array(training_input)
input_file.close()

# reading train labels from file

input_file = open("data/train_labels.txt", "rt", encoding="utf-8")
training_labels = input_file.read().splitlines()
for i in range(len(training_labels)):
    training_labels[i] = training_labels[i].split("\t")[1]
training_labels = np.array(training_labels, dtype=int)
input_file.close()

# reading validation input from file

input_file = open("data/validation_samples.txt", "rt", encoding="utf-8")
validation_inputs = input_file.read().splitlines()
for i in range(len(validation_inputs)):
    validation_inputs[i] = validation_inputs[i].split("\t")[1]
validation_inputs = np.array(validation_inputs)
input_file.close()

# reading validation labels from file

input_file = open("data/validation_labels.txt", "rt", encoding="utf-8")
validation_labels = input_file.read().splitlines()
for i in range(len(validation_labels)):
    validation_labels[i] = validation_labels[i].split("\t")[1]
validation_labels = np.array(validation_labels, dtype=int)
input_file.close()


# reading test inputs

test_Ids = []

input_file = open("data/test_samples.txt", "rt", encoding="utf-8")
test_inputs = input_file.read().splitlines()
for i in range(len(test_inputs)):
    test_Ids.append(test_inputs[i].split("\t")[0])
    test_inputs[i] = test_inputs[i].split("\t")[1]
test_inputs = np.array(test_inputs)
input_file.close()

# convert string input to vectors, using features of sklearn library
vectorizer = CountVectorizer(analyzer="char", ngram_range=(1, 1))
#vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 1), max_df=0.7)
vectorizer.fit(training_input)


training_input_vectorized = vectorizer.transform(training_input)
validation_inputs_vectorized = vectorizer.transform(validation_inputs)
test_inputs_vectorized = vectorizer.transform(test_inputs)

### CATEGORICAL NATIVE BAYES
NBModel1 = BernoulliNB()
#NBModel1.fit(training_input_vectorized, training_labels)
#prediction1 = NBModel1.predict(test_inputs_vectorized)
#precision = NBModel1.score(validation_inputs_vectorized, validation_labels)

### MULTINOMIAL NATIVE BAYES
NBModel2 = MultinomialNB()
#NBModel2.fit(training_input_vectorized, training_labels)
#prediction2 = NBModel2.predict(test_inputs_vectorized)
#precision = NBModel2.score(validation_inputs_vectorized, validation_labels)

### COMPLEMENTAL NATIVE BAYES
NBModel3 = ComplementNB()
#NBModel3.fit(training_input_vectorized, training_labels)
#prediction3 = NBModel3.predict(test_inputs_vectorized)
#precision = NBModel3.score(validation_inputs_vectorized, validation_labels)

### SUPPORT VECTOR MACHINES

SVCModel = SVC(kernel='linear')
SVCModel.fit(training_input_vectorized, training_labels)
#prediction4 = SVCModel.predict(test_inputs_vectorized)
#precision = SVCModel.score(validation_inputs_vectorized, validation_labels)


#print(precision)

# final_prediction = []
# for i in range(len(prediction1)):
#     votes = [0, 0, 0, 0]
#     votes[prediction1[i]] += 1
#     votes[prediction2[i]] += 1
#     votes[prediction3[i]] += 1
#
#     max_votes = max(votes[1], votes[2], votes[3])
#
#     if max_votes == votes[1]:
#         final_prediction.append(1)
#     elif max_votes == votes[2]:
#         final_prediction.append(2)
#     else:
#         final_prediction.append(3)

# prediction = prediction3
#
# # output to some file
#
# output_file = open("output/output_file.txt", "w", encoding="utf-8")
# output_file.write("id,label\n")
#
# for i in range(len(prediction)):
#    output_file.write(test_Ids[i] + ',' + str(prediction[i]))
#    if i < len(prediction) - 1:
#        output_file.write('\n')
#
# output_file.close()



