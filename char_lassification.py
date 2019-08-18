import glob

all_filenames = glob.glob('data/names/*.txt')
print(all_filenames)



#training on GPU
 
import torch   
#define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import unicodedata
import string

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)



# Turn a Unicode string to plain ASCII
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# Build the category_lines dictionary, a list of names per language
category_words = {}
all_categories = []

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding="utf8").read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]

for filename in all_filenames:
    category = filename.split('\\')[1].split('.')[0]
    all_categories.append(category)
    words = readLines(filename)
    category_words[category] = words

n_categories = len(all_categories)


import torch

def letter_to_tensor(letter):
    tensor = torch.zeros(1, n_letters)
    letter_index = all_letters.find(letter)
    tensor[0][letter_index] = 1
    return tensor

def word_to_tensor(word):
    tensor = torch.zeros(len(word), 1, n_letters)
    for li, letter in enumerate(word):
        letter_index = all_letters.find(letter)
        tensor[li][0][letter_index] = 1
    return tensor

# Creating the Network

import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax()
    
    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))


n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)

rnn.to(device)


def category_from_output(output):
    top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
    category_i = top_i[0][0]
    return all_categories[category_i], category_i



import random

def random_training_pair():                                                                                                               
    category = random.choice(all_categories)
    word = random.choice(category_words[category])
    category_tensor = Variable(torch.LongTensor([all_categories.index(category)])).to(device)
    word_tensor = Variable(word_to_tensor(word)).to(device)
    return category, word, category_tensor, word_tensor

for i in range(10):
    category, word, category_tensor, word_tensor = random_training_pair()
    print('category =', category, '/ word =', word)


# Training the Network

criterion = nn.NLLLoss()

learning_rate = 0.005 
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)


def train(category_tensor, word_tensor):
    rnn.zero_grad()
    hidden = rnn.init_hidden().to(device)
    
    for i in range(word_tensor.size()[0]):
        output, hidden = rnn(word_tensor[i], hidden)
        output.to(device)

    loss = criterion(output, category_tensor)
    loss.backward()

    optimizer.step()

    return output, loss.data[0]


import time
import math

n_epochs = 100000
print_every = 5000
plot_every = 1000

# Keep track of losses for plotting
current_loss = 0
all_losses = []

def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)



start = time.time()

for epoch in range(1, n_epochs + 1):
    # Get a random training input and target
    category, word, category_tensor, word_tensor = random_training_pair()
    output, loss = train(category_tensor, word_tensor)
    
    current_loss += loss
    
    # Print epoch number, loss, name and guess
    if epoch % print_every == 0:
        guess, guess_i = category_from_output(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (epoch, epoch / n_epochs * 100, time_since(start), loss, word, guess, correct))

    # Add current loss avg to list of losses
    if epoch % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0


# Plotting the Results

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_losses)


# Evaluating the Results

confusion = torch.zeros(n_categories, n_categories).to(device)
n_confusion = 10000

# Just return an output given a line
def evaluate(word_tensor):
    hidden = rnn.init_hidden().to(device)
    
    for i in range(word_tensor.size()[0]):
        output, hidden = rnn(word_tensor[i], hidden)
    
    return output.to(device)

# Go through a bunch of examples and record which are correctly guessed
for i in range(n_confusion):
    category, word, category_tensor, word_tensor = random_training_pair()
    output = evaluate(word_tensor)
    guess, guess_i = category_from_output(output)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] += 1

# Normalize by dividing every row by its sum
for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.cpu().numpy())
fig.colorbar(cax)

# Set up axes
ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

plt.show()

def predict(input_word, n_predictions=3):
    print('\n> %s' % input_word)
    output = evaluate(Variable(word_to_tensor(input_word)).to(device))

    # Get top N categories
    topv, topi = output.data.topk(n_predictions, 1, True)
    predictions = []

    for i in range(n_predictions):
        value = topv[0][i]
        category_index = topi[0][i]
        print('(%.2f) %s' % (value, all_categories[category_index]))
        predictions.append([value, all_categories[category_index]])

predict('Dovesky')
predict('Jackson')
predict('Satoshi')
