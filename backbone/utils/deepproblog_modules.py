import torch
from torch import nn
from problog.evaluator import Semiring

class GraphSemiring(Semiring):

    def __init__(self, batch_size=32, device=torch.device('cpu')):
        Semiring.__init__(self)
        self.eps = 1e-12
        self.batch_size = batch_size
        self.device = device

    def negate(self, a):
        """Returns the negation of the probability a: 1-a."""
        return self.one() - a

    def one(self):
        """Returns the identity element of the multiplication."""
        one = torch.ones(self.batch_size)
        return one.to(self.device)

    def zero(self):
        """Returns the identity element of the addition."""
        zero = torch.zeros(self.batch_size)
        return zero.to(self.device)

    def is_zero(self, a):
        """Tests whether the given value is the identity element of the addition up to a small constant"""
        return ((a >= -self.eps) & (a <= self.eps)).any()

    def is_one(self, a):
        """Tests whether the given value is the identity element of the multiplication up to a small constant"""
        return ((a >= 1.0 - self.eps) & (a <= 1.0 + self.eps)).any()

    def plus(self, a, b):
        """Computes the addition of the given values."""
        if self.is_zero(b):
            return a
        if self.is_zero(a):
            return b
        return a + b

    def times(self, a, b):
        """Computes the multiplication of the given values."""
        if self.is_one(b):
            return a
        if self.is_one(a):
            return b
        return a * b

    def set_weights(self, weights):
        self.weights = weights

    def normalize(self, a, z):
        return a / z

    def value(self, a):
        """Transform the given external value into an internal value."""
        v = self.weights.get(a, a)
        return v


class DeepProblogModel(nn.Module):

    def __init__(self, encoder, model_dict, n_facts=20, nr_classes=19, dropout=0.5, is_train=True, device='cpu'):
        super(DeepProblogModel, self).__init__()
        self.encoder = encoder
        self.dropout = dropout
        self.model_dict = model_dict  # Dictionary of pre-compiled ProbLog models
        self.is_train = is_train
        self.device = device
        self.nr_classes = nr_classes
        # Herbrand base
        self.herbrand_base = self.define_herbrand_base(n_facts).to(self.device)
        # Weights dictionary for problog inference
        self.weights_dict = self.build_weights_dictionary(n_facts)

    def forward(self, x):
        # Image encoding
        z = self.encoder(x)
        # normalize concept preditions
        self.facts_probs = self.normalize_concepts(z)
        # Problog inference to compute worlds and query probability distributions
        self.query_prob, self.worlds_prob = self.problog_inference(self.facts_probs)

        return self.query_prob,self.facts_probs

    def herbrand(self, world):
        """Herbrand representation of the given world(s)"""
        return torch.matmul(world, self.herbrand_base)

    def define_herbrand_base(self, n_facts):
        """Defines the herbrand base to encode ProbLog worlds"""
        pass

    def build_weights_dictionary(self, n_facts):
        """Returns the weights dictionary used during ProbLog inference to update the graph semiring."""
        pass

    def normalize_concepts(self, z):
        """Computes the probability for each ProbLog fact given the latent vector z"""
        pass

    def problog_inference(self, facts_probs, labels=None, query=None):
        """
        Performs ProbLog inference to retrieve the worlds probability distribution P(w) and the desired query probability.
        """
        # Update weights of graph semiring with the facts probability
        self.update_semiring_weights(facts_probs)
        # Select pre-compiled ProbLog model corresponding to the query
        sdd = self.model_dict['query'][labels]
        # Evaluate model
        res = sdd.evaluate(semiring=self.semiring)
        # Extract query probability
        query_prob = self.extract_query_probability(res)
        # Extract worlds probability P(w)
        self.worlds_prob = self.extract_worlds_probability(res)
        return query_prob, self.worlds_prob

    def problog_inference_with_evidence(self, facts_probs, evidence):
        """
        Performs ProbLog inference to retrieve the worlds probability distribution P(w) given the evidence.
        """
        # Update weights of graph semiring with the facts probability
        self.update_semiring_weights(facts_probs)
        # Select pre-compiled ProbLog model corresponding to the evidence
        sdd = self.model_dict['evidence'][evidence]
        # Evaluate model
        res = sdd.evaluate(semiring=self.semiring)
        # Extract worlds probability P(w)
        worlds_prob = self.extract_worlds_probability(res)
        return worlds_prob

    def update_semiring_weights(self, facts_probs):
        """Updates weights of graph semiring with the facts probability"""
        pass

    def extract_worlds_probability(self, res):
        """Extracts P(q) contained in the dictionary 'res' resulting from ProbLog model evaluation."""
        pass

    def extract_query_probability(self, res):
        """Extracts P(w) contained in the dictionary 'res' resulting from ProbLog model evaluation."""
        pass
