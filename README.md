# chatbot
Used Pytorch to succesfully create a chatbot that uses a corpus composed of movie dialogues to train the model.

# Formatting the data file

For starters, each line of the datafile is split into a dictionary of fields present in the file. The fields of lines are then grouped based on how they appear in the conversations dataset. Sentence pairs are then extracted and appended together.

# Loading and trimming the data

For this project, we deal with sequences of words that do not map to a discrete numerical space. This means that we need to map each unique word to a numerical value before we can proceed.

To make our job easier, a Voc class was defined that keeps track of the mapping from words to indexes, the corresponding reverse mapping, a count of each word and the total word count. Next, we perform trimming and loading by converting the strings into plain ASCII, performing pre-processing and then calling the Voc class to read query and response pairs and return a voc object. Also, rarely used words are trimmed  to achieve faster convergence (convergence is the point where additional training cannot improve the model).

# Prepping the data

The model uses numerical torch tensors (multi-dimensional matrix containing a single datatype) so the data must be prepared and converted into tensors. To increase the speed of training and utilize GPU capabilities, the data is split into min-batches.

When mini-batches are used, the variation of sentence length is something that must be taken into account. To accomodate sentences of different sizes, the input tensors that are shorter than the maximum length are zero padded. The input batch shape is then transposed so indexing across the first dimension returns a time step across all sentences.

# Seq2Seq model

By using two separate neural networks (RNN), where one acts as an encoder that encodes a variable length input sequence to fixed-length vector. The second RNN is a decoder that takes an input word and the context vector, and returns a possible next word along with a hidden state for the next iteration.

# Masked loss

Due to the zero padding, we cannot consider all elements of the tensor whne we calculate loss. The function helps to calculate the loss based on the decoders output tensor, the target tensor and a binary mask tensor. The loss function calculates the negative log likelihood of the elements that correpsond to a 1.

# Sequence of operations

  Forward pass entire input batch through encoder.
  
  Initialize decoder inputs as SOS_token, and hidden state as the encoder’s final hidden state.
  
  Forward input batch sequence through decoder one time step at a time.
  
  If teacher forcing: set next decoder input as the current target; else: set next decoder input    as current decoder output.
  
  Calculate and accumulate loss.
  
  Perform backpropagation.
  
  Clip gradients.
  
  Update encoder and decoder model parameters.
