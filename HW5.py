import torch
from torchtext import datasets

torch.manual_seed(1)

# загрузка датасета
conll_train_iter, conll_test_iter = datasets.CoNLL2000Chunking()

word_to_ix = {}
tag_to_ix = {}
for sent, tags, chunks in conll_train_iter:
    for word in sent:
        if word.lower() not in word_to_ix:
            word_to_ix[word.lower()] = len(word_to_ix)

    for tag in tags:
        if tag.lower() not in tag_to_ix:
            tag_to_ix[tag.lower()] = len(tag_to_ix)


EMBEDDING_DIM = 100
HIDDEN_DIM = 100


class LSTMTagger(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_layer = torch.nn.Embedding(len(word_to_ix), EMBEDDING_DIM)
        self.lstm = torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM)

        self.pos_predictor = torch.nn.Linear(HIDDEN_DIM, len(tag_to_ix))

    def forward(self, token_ids):
        embeds = self.embedding_layer(token_ids)
        lstm_out, _ = self.lstm(embeds.view(len(token_ids), 1, -1))
        logits = self.pos_predictor(lstm_out.view(len(token_ids), -1))
        probs = torch.nn.functional.softmax(logits, dim=1)

        return probs


model = LSTMTagger()
num_epoch = 10
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w.lower()] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


for epoch in range(num_epoch):
    error = 0
    for v in conll_train_iter:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        sentence_in = prepare_sequence(sent, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        error += loss
        loss.backward()
        optimizer.step()


# See what the scores are after training
tag_scores = []
for sent, tags, chunks in conll_train_iter:
  with torch.no_grad():
    inputs = prepare_sequence(sent, word_to_ix)
    tag_scores.append(model(inputs))