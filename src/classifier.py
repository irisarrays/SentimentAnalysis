import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn.functional import softmax
import torch.nn as nn
from collections import Counter
from tqdm import tqdm
import spacy
import time
import datetime

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
nlp = spacy.load("en_core_web_sm")


class Classifier:

    def __init__(self, train_batch_size=12, eval_batch_size=8, max_length=128, lr=2e-5, eps=1e-5, n_epochs=10):
        """
        :param train_batch_size: (int) Training batch size
        :param eval_batch_size: (int) Batch size while using the `predict` method.
        :param max_length: (int) Maximum length for padding
        :param lr: (float) Learning rate
        :param eps: (float) Adam optimizer epsilon parameter
        :param n_epochs: (int) Number of epochs to train
        """
        # model parameters
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.max_length = max_length
        self.lr = lr
        self.eps = eps
        self.n_epochs = n_epochs

        # Information to be set or updated later
        self.trainset = None
        self.categories = None
        self.labels = None
        self.model = None

        # Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # define the pretrained model
        configs = BertConfig.from_pretrained('bert-base-uncased', num_labels=3)  # BERT configuration
        self.model = BertForSequenceClassification(configs)
        self.model.to(device)

    def train(self, trainfile, devfile):

        """Trains the classifier model on the training set stored in file trainfile"""

        # Loading the data and splitting up its information in lists
        trainset = np.genfromtxt(trainfile, delimiter='\t', dtype=str, comments=None)
        self.trainset = trainset

        n = len(trainset)

        # get the opinions
        targets = trainset[:, 0]
        self.labels = list(Counter(targets).keys())

        # get the aspects
        categories = trainset[:, 1]
        self.categories = list(Counter(categories).keys())

        # get the reviwed objects
        objects = trainset[:,2]

        # get the words from start to end
        start_end = [[int(x) for x in w.split(':')] for w in trainset[:, 3]]
        words = [trainset[:, 4][i][start_end[i][0]:start_end[i][1]] for i in range(n)]

        # get the complete reviews
        sentences = [str(s) for s in trainset[:, 4]]

        # Tokenization
        attention_masks = []
        input_ids = []
        token_type_ids = []
        labels = []
        for category, obj, word, sentence in zip(categories, objects, words, sentences):
            encoded_dict = self.tokenizer.encode_plus(category, obj+' '+word.lower()+ ' ' + sentence.lower(),
                                                      add_special_tokens=True,  # Add '[CLS]' and '[SEP]' tokens
                                                      max_length=self.max_length,  # Pad & truncate all sequences
                                                      pad_to_max_length=True,
                                                      return_attention_mask=True,  # Construct attention masks
                                                      return_tensors='pt',  # Return pytorch tensors.
                                                      )
            attention_masks.append(encoded_dict['attention_mask'])
            input_ids.append(encoded_dict['input_ids'])
            token_type_ids.append(encoded_dict['token_type_ids'])
        attention_masks = torch.cat(attention_masks, dim=0)
        input_ids = torch.cat(input_ids, dim=0)
        token_type_ids = torch.cat(token_type_ids, dim=0)

        # Converting polarities into integers (0: positive, 1: negative, 2: neutral)
        for target in targets:
            if target == 'positive':
                labels.append(0)
            elif target == 'negative':
                labels.append(1)
            elif target == 'neutral':
                labels.append(2)
        labels = torch.tensor(labels)

        # Pytorch data iterators
        train_data = TensorDataset(input_ids, attention_masks, token_type_ids, labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data,
                                      batch_size=self.train_batch_size,
                                      sampler=train_sampler)

        # Optimizer and scheduler
        no_decay = ['bias', 'gamma', 'beta']

        optimizer_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
         # In Transformers, optimizer and schedules are split and instantiated like this:
        optimizer = AdamW(optimizer_parameters,
                          lr=self.lr,
                          eps=self.eps)

        # optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)

        total_steps = len(train_dataloader) * self.n_epochs

        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=total_steps)

        initial_t0 = time.time()
        for epoch in range(self.n_epochs):
            print('\n   ======== Epoch %d / %d ========' % (epoch + 1, self.n_epochs))
            print('   Training...\n')

            total_train_loss = 0

            self.model.train()

            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)

                input_ids_, input_mask_, segment_ids_, label_ids_ = batch

                self.model.zero_grad()

                # compute the loss
                loss = self.model(input_ids_,
                                     token_type_ids=segment_ids_,
                                     attention_mask=input_mask_,
                                     labels=label_ids_).loss

                # Accumulate the training loss over all of the batches
                # so that we can calculate the average loss at the end.
                total_train_loss += loss.item()

                # Perform a backward pass to calculate the gradients.
                loss.backward()

                # clip gradient norm to 1.0
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()

                scheduler.step()

            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(train_dataloader)
            # print("     Average training loss: {0:.2f}".format(avg_train_loss))

    def predict(self, datafile):

        """
        Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """

        # Loading the data and splitting up its information in lists
        evalset = np.genfromtxt(datafile, delimiter='\t', dtype=str, comments=None)
        m = len(evalset)

        # get categories
        categories = evalset[:, 1]

        # get the reviwed objects
        objects = evalset[:,2]

        # get words from start to end
        start_end = [[int(x) for x in w.split(':')] for w in evalset[:, 3]]
        words = [evalset[:, 4][i][start_end[i][0]:start_end[i][1]] for i in range(m)]

        # get the complete review sentences
        sentences = [str(s) for s in evalset[:, 4]]

        # Tokenization
        attention_masks = []
        input_ids = []
        token_type_ids = []
        for category, obj, word, sentence in zip(categories, objects, words, sentences):
            encoded_dict = self.tokenizer.encode_plus(category, obj+' '+word.lower()+' '+sentence.lower(),
                                                      add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                                                      max_length=self.max_length,  # Pad & truncate all sequences
                                                      pad_to_max_length=True,
                                                      return_attention_mask=True,  # Construct attention masks
                                                      return_tensors='pt',  # Return pytorch tensors.
                                                      )
            attention_masks.append(encoded_dict['attention_mask'])
            input_ids.append(encoded_dict['input_ids'])
            token_type_ids.append(encoded_dict['token_type_ids'])
        attention_masks = torch.cat(attention_masks, dim=0)
        input_ids = torch.cat(input_ids, dim=0)
        token_type_ids = torch.cat(token_type_ids, dim=0)

        # Pytorch data iterators
        eval_data = TensorDataset(input_ids, attention_masks, token_type_ids)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data,
                                     batch_size=self.eval_batch_size,
                                     sampler=eval_sampler)

        # Prediction
        named_labels = []

        self.model.eval()

        for batch in eval_dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids = batch

            with torch.no_grad():
                logits = self.model(input_ids,
                                    token_type_ids=segment_ids,
                                    attention_mask=input_mask)[0]

            logits = softmax(logits, dim=-1)
            logits = logits.detach().cpu().numpy()
            outputs = np.argmax(logits, axis=1)

            # converting integer labels into named labels
            for label in outputs:
                if label == 0:
                    named_labels.append('positive')
                elif label == 1:
                    named_labels.append('negative')
                elif label == 2:
                    named_labels.append('neutral')

        return np.array(named_labels)
