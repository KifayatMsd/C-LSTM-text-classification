# Multi-class Text Classification
Implemention of C-LSTM in Tensorflow for multi-class text classification problem.
## Models
* The proposed C-LSTM model given in below paper is used for text classification.
* Paper: A C-LSTM Neural Network for Text Classification
* Link : https://arxiv.org/abs/1511.08630).
## Requirements  
* Python 3.x  
* Tensorflow > 1.5
* Sklearn > 0.19.0  
## Data Format
Training data should be stored in csv file. The first line of the file should be ["label", "content"] or ["content", "label"].
## Train
Run train.py to train the models.
Parameters:
```
optional arguments:
  --clf CLF             
                        Type of classifiers. i.e. clstm
  --data_file DATA_FILE
                        Data file path
  --stop_word_file STOP_WORD_FILE
                        Stop word file path
  --language LANGUAGE   Language of the data file. You have two choices: [ch,
                        en]
  --min_frequency MIN_FREQUENCY
                        Minimal word frequency
  --num_classes NUM_CLASSES
                        Number of classes
  --max_length MAX_LENGTH
                        Max document length
  --vocab_size VOCAB_SIZE
                        Vocabulary size
  --test_size TEST_SIZE
                        Cross validation test size
  --embedding_size EMBEDDING_SIZE
                        Word embedding size. For CNN, C-LSTM.
  --filter_sizes FILTER_SIZES
                        CNN filter sizes. For CNN, C-LSTM.
  --num_filters NUM_FILTERS
                        Number of filters per filter size. For CNN, C-LSTM.
  --hidden_size HIDDEN_SIZE
                        Number of hidden units in the LSTM cell. For LSTM, Bi-
                        LSTM
  --num_layers NUM_LAYERS
                        Number of the LSTM cells. For LSTM, Bi-LSTM, C-LSTM
  --keep_prob KEEP_PROB
                        Dropout keep probability
  --learning_rate LEARNING_RATE
                        Learning rate
  --l2_reg_lambda L2_REG_LAMBDA
                        L2 regularization lambda
  --batch_size BATCH_SIZE
                        Batch size
  --num_epochs NUM_EPOCHS
                        Number of epochs
  --decay_rate DECAY_RATE
                        Learning rate decay rate. Range: (0, 1]
  --decay_steps DECAY_STEPS
                        Learning rate decay steps.
  --evaluate_every_steps EVALUATE_EVERY_STEPS
                        Evaluate the model on validation set after this many
                        steps
  --save_every_steps SAVE_EVERY_STEPS
                        Save the model after this many steps
  --num_checkpoint NUM_CHECKPOINT
                        Number of models to store
```
You could run train.py to start training. For example:
```
python train.py --data_file=./data/data.csv --clf=clstm
```

After the training is done, you can use tensorboard to see the visualizations of the graph, losses and evaluation metrics:  

```
tensorboard --logdir=./runs/1111111111/summaries
```

## Test 
Run test.py to evaluate the trained model  
Parameters: 
```
optional arguments:
  --test_data_file TEST_DATA_FILE
                        Test data file path
  --run_dir RUN_DIR     Restore the model from this run
  --checkpoint CHECKPOINT
                        Restore the graph from this checkpoint
  --batch_size BATCH_SIZE
                        Test batch size
```
You could run test.py to start evaluation. For example:
```
python test.py --test_data_file=./data/data.csv --run_dir=./runs/1111111111 --checkpoint=clf-10000
```
