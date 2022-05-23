import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

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

# MultinomialNB expects numerical values, so we
# convert string input to vectors, using features of sklearn library
vectorizer = CountVectorizer()
vectorizer.fit(training_input)


training_input_vectorized = vectorizer.transform(training_input)
validation_inputs_vectorized = vectorizer.transform(validation_inputs)
test_inputs_vectorized = vectorizer.transform(test_inputs)

### MULTINOMIAL NATIVE BAYES

NBModel = MultinomialNB()
NBModel.fit(training_input_vectorized, training_labels)
prediction = NBModel.predict(test_inputs_vectorized)
#precision = NBModel.score(validation_inputs_vectorized, validation_labels)

output_file = open("output/output_file.txt", "w", encoding="utf-8")

output_file.write("id,label\n")

for i in range(len(prediction)):
   output_file.write(test_Ids[i] + ',' + str(prediction[i]))
   if i < len(prediction) - 1:
       output_file.write('\n')

output_file.close()



