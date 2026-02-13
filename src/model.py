import torch

class BOWVectorizer:
    def __init__(self, train_text):
        # Note train_text is a list of strings, where each string is a training example
        self.train_text = train_text
        self.vocab = None
        self.word2idx = None
    
    def fit(self):
        # Create vocab of each character in the training text
        self.vocab = set(self.train_text)
        self.vocab = sorted(list(self.vocab))
        self.char2idx = {char: idx for idx, char in enumerate(self.vocab)}
    
    def transform(self, text):
        if self.vocab is None:
            raise ValueError("Vocabulary not built. Call fit() first.")

        train_features = torch.zeros((len(text), len(self.vocab)), dtype=torch.float32)
        for i, sent in enumerate(text): 
            for char in sent:
                if char in self.char2idx:
                    train_features[i, self.char2idx[char]] += 1

        return train_features