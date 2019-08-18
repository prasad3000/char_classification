# char_classification
 character level classification using rnn

## Getting Started

We will be building and training a basic character-level RNN to classify words. A character-level RNN reads words as a series of characters - outputting a prediction and "hidden state" at each step, feeding its previous hidden state into each next step. We take the final prediction to be the output, i.e. which class the word belongs to.

Specifically, we'll train on a few thousand surnames from 18 languages of origin, and predict which language a name is from based on the spelling:

## Preparing the Data
 
Included in the `data/names` directory are 18 text files named as "[Language].txt". Each file contains a bunch of names, one name per line, mostly romanized (but we still need to convert from Unicode to ASCII).
 
We'll end up with a dictionary of lists of names per language, `{language: [names ...]}`. The generic variables "category" and "line" (for language and name in our case) are used for later extensibility.

To represent a single letter, we use a "one-hot vector" of size `<1 x n_letters>`. A one-hot vector is filled with 0s except for a 1 at index of the current letter, e.g. `"b" = <0 1 0 0 0 ...>`.

To make a word we join a bunch of those into a 2D matrix `<line_length x 1 x n_letters>`.

The '1' represent the batch.

## Model

To design our RNN model we initialized the hidden as random and combined it with input to fed into a liner network the final hidden is the output and we again apply softmax to the output to classify.

After that we are converting all nd-array to tensor. And copy them to cpu or cuda to run them on cpu or gpu.

To run a step of this network we need to pass an input (in our case, the Tensor for the current letter) and a previous hidden state (which we initialize as zeros at first). We'll get back the output (probability of each language) and a next hidden state (which we keep for the next step).

Here we use variable because PyTorch modules operate on Variables rather than straight up Tensors

## Output

The softmax will give the probability of a category, from them we pick the highest probable index and heance highest probable category.

## Training

Here we are using NLLLose you can read https://pytorch.org/docs/stable/nn.html#nllloss for more details

learning_rate is medimum; If you set this too high, it might explode. If too low, it might not learn. we will make it 0.005

Here we use SGD optimizer read https://pytorch.org/docs/stable/optim.html#how-to-use-an-optimizer for more details

# Each loop of training will:

1. Create input and target tensors
2. Create a zeroed initial hidden state
3. Read each letter in and
     * Keep hidden state for next letter
4. Compare final output to target
5. Back-propagate
6. Return the output and loss

## Evaluate and confussion matrix

1. Create a zeroed initial hidden state
2. Read each letter in and
     * Keep hidden state for next letter
3. From the output select the top category
4. Increase the confussion matrix based on the correct prediction
5. Normalize the confussion matrix by dividing every row by its sum
6. Plot them using matshow


## Prediction

It will predict first k number of possiblity with their value.

## What next

it can be used for any classifier like
* Any word -> language
* First name -> gender
* Character name -> writer
* Page title -> blog or subreddit
