# Standford Sentiment Treebank


- [Objective & Data description](#objective--data-description)
- [Proposed Solution](#proposed-solution)
- [Augmentation](#augmentation)
- [Model and Loss function](#model-and-loss-function)
- [Results: Accuracy, Loss](#results-accuracy-loss)
- [Further Improvements](#further-improvements)


## Objective & Data description
The Stanford Sentiment Treebank dataset  consists of 11,855 sentences extracted from movie reviews with fine-grained sentiment labels [1–5] (1 most negative and 5 most positive). It also consists of well as 215,154 phrases that compose each sentence in the dataset. This was also to be achieved by using various NLP data augmentation techniques like "random_insertion", "Back Translate", "random_swap" and "random_delete".

## Proposed Solution
Eventhough we have 8545 sentences, still we augment the data using multiple augmentation techniques to increase our data-size to train our models better. For the model, we use a multi-layer LSTM model fed by an embedding layer. To classify the sentences into their perceived sentiment labels from the LSTM layer, we use two fully connected layers with output dimension equal to both the sentiments' labels present in the dataset. 

![](LSTM_Network.png)

## Augmentation
Data augmentation in data analysis are techniques used to increase the amount of data by adding slightly modified copies of already existing data or newly created synthetic data from existing data. It acts as a regularizer and helps reduce overfitting when training a machine learning model.
We have used 4 data augmentation techniques:
- **random_insertion**: we randomly insert synonyms of a word at a random position in the sentence
- **random_deletion**: we randomly delete a word from the sentence. This is achieved by generating a random number from a uniform distribution for each of the words in the sentence. This generated number between 0 and 1, and the words smaller than than the pre-defined threshold for every sentence. This allows for a random deletion of some words from the sentence
- **random_swap**: we randomly swap the order of two words in a sentence. To achieve this, we smaple two indices (index1 != index2; index1 & index2 < length of sentence > 1)
- **translate_and_back**: we take a sentence and convert it to a language randomly chosen from the **_google_trans_new_** library's list of languages and then translate it back to english, which is the original language of our tweet. We use translate only 4% of the times to limit the API calls to google server and avoid bad_request timeouts

We augment each of the tweet 6 times, **randomly** choosing one of the 4 augmentation techniques mentioned above with probabilities of 33% each for the first three and 1% for translate_and_back. We have kept our augmentation per tweet to 6 times so as to reduce our data augmentation runtime. This can be exceeded to 16 based on the paper[^1].



## Model and Loss function
As discussed in proposed architecture, we use a model with embedding (600), 3 LSTM and 2 fully connected layers.  First we pass our augmented and original dataset through _spacy_ to tokenize it. We tokenize the training, validation and testing daya individually as our dataset is already split into train-valid-test hence we teh same split to create the bucketiterators. Using train dataset we created a vocabulary.

We use _Adam_ optimiser with a learning rate of 1*10^-4 and _CrossEntropyLoss_ 

![](cross_entropy.png)

Cross entropy loss is used since  it is a multi-class classification problem.
This model was trained for 25 epochs. 
```
Epoch 1 | Time Taken: 20.04s
	Train Loss  5 labels: 1.573 | Train Acc  5 labels: 29.51%
	 Val. Loss  5 labels: 1.555 |  Val. Acc  5 labels: 32.48% 

Epoch 2 | Time Taken: 19.88s
	Train Loss  5 labels: 1.512 | Train Acc  5 labels: 38.61%
	 Val. Loss  5 labels: 1.536 |  Val. Acc  5 labels: 34.90% 

Epoch 3 | Time Taken: 19.98s
	Train Loss  5 labels: 1.460 | Train Acc  5 labels: 44.09%
	 Val. Loss  5 labels: 1.530 |  Val. Acc  5 labels: 35.10% 
.....	 
.....	 
.....
Epoch 38 | Time Taken: 19.92s
	Train Loss  5 labels: 0.981 | Train Acc  5 labels: 92.97%
	 Val. Loss  5 labels: 1.572 |  Val. Acc  5 labels: 31.99% 

Epoch 39 | Time Taken: 19.98s
	Train Loss  5 labels: 0.978 | Train Acc  5 labels: 93.18%
	 Val. Loss  5 labels: 1.576 |  Val. Acc  5 labels: 31.57% 

Epoch 40 | Time Taken: 19.95s
	Train Loss  5 labels: 0.975 | Train Acc  5 labels: 93.52%
	 Val. Loss  5 labels: 1.572 |  Val. Acc  5 labels: 32.15% 
	 
```
As we can see, the model for sentiments [1-5] starts from 32.48% validation accuracy and 1.555 validation loss which by 40th epoch is decreased to  Val. Loss  5 labels: 1.572 |  Val. Acc   5 labels: 32.15%


## Results: Accuracy, Loss
Post training our model for sentiments' labels [1-5], we get a training accuracy of 96.44% and a validation accuracy of 97.63%.

The training and validation loss the model for sentiments_5 (for sentiments on scale 1-5):

![](train_valid_label_loss_5.png)

The training and validation accuracy the model for sentiments_5 (for sentiments on scale 1-5):

![](train_valid_label_accuracy_5.png)

Confusion Matrix:


![](confusion_matrix.png)



## Further Improvements

- As we saw in the confusion matrix, our model makes most classification mistakes in the positive classes. Hence, special notice and augmentation for the positive classes may help in lifting the score
- We only use an simple LSTM model here. More sophisticated models can be used for better prediction
- We see this as a case of overfitting as the validation accuracy keeps decreasing futher from the increasing train accuracy
- This time we did did it only for 5 classes. Training can also be extended to 1-25 classes to see any performance changes



[^1]  **EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks** (https://arxiv.org/abs/1901.11196)


## Group

- Anirban Mukherjee
- Debasish Sarangi










