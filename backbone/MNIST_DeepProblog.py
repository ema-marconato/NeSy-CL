import torch
from problog.logic import Constant, Term
from torch import nn
from backbone.utils.encoders import PairCNNEncoder
from backbone.utils.deepproblog_modules import GraphSemiring
from backbone.utils.deepproblog_modules import DeepProblogModel
from backbone.utils.utils_problog import build_worlds_queries_matrix

class MNIST_DeepProblog(DeepProblogModel):
    def __init__(self, encoder= None,
                 model_dict=None,
                 n_facts=20, nr_classes=19, dropout=0.5,
                 is_train=True, device="cuda"):
        encoder=PairCNNEncoder(hidden_channels=64, latent_dim=20,dropout=0.5) # PairMLPEncoder(28*28)
        #model_dict= build_model_dict(2, 10)
        super(MNIST_DeepProblog, self).__init__(encoder=encoder, model_dict=model_dict,
                                                n_facts=n_facts, nr_classes=nr_classes, dropout=dropout,
                                                is_train=is_train, device=device)

        self.n_facts=n_facts
        self.w_q = build_worlds_queries_matrix(2, 10)
        self.w_q = self.w_q.to(device)  # Worlds-queries matrix

    def normalize_concepts(self, z):
        """Computes the probability for each ProbLog fact given the latent vector z"""
        # Extract probs for each digit
        prob_digit1, prob_digit2 = z

        # Add stochasticity on prediction
        prob_digit1 += 1 * torch.randn_like(prob_digit1, device=prob_digit1.device)
        prob_digit2 += 1 * torch.randn_like(prob_digit2, device=prob_digit1.device)

        # Sotfmax on digits_probs (the 10 digits values are mutually exclusive)
        prob_digit1 = nn.Softmax(dim=1)(prob_digit1)
        prob_digit2 = nn.Softmax(dim=1)(prob_digit2)

        # Clamp digits_probs to avoid ProbLog underflow
        eps = 1e-5
        prob_digit1 = prob_digit1 + eps
        with torch.no_grad():
            Z1 = torch.sum(prob_digit1, dim=-1, keepdim=True)
        prob_digit1 = prob_digit1 / Z1  # Normalization
        prob_digit2 = prob_digit2 + eps
        with torch.no_grad():
            Z2 = torch.sum(prob_digit2, dim=-1, keepdim=True)
        prob_digit2 = prob_digit2 / Z2  # Normalization

        return torch.stack([prob_digit1, prob_digit2], dim=1)

    def define_herbrand_base(self, n_facts):
        """Defines the herbrand base to encode ProbLog worlds"""
        n_digits = n_facts // 2
        eye = torch.eye(n_digits)
        herbrand_base = torch.cat(
            [torch.cat((eye[i].expand(n_digits, n_digits), eye), dim=1) for i in
             range(n_digits)], dim=0)

        return herbrand_base

    def build_weights_dictionary(self, n_facts):
        """Returns the weights dictionary used during ProbLog inference to update the graph semiring."""
        n_digits = n_facts // 2
        weights_dict = {}
        for i in range(1, 3):
            for j in range(n_digits):
                key = 'p_' + str(i) + str(j)
                weights_dict[Term(key)] = "NOT DEFINED"

        return weights_dict

    def compute_query(self, query, worlds_prob):
        """Computes query probability given the worlds probability P(w)."""
        # Select the column of w_q matrix corresponding to the current query
        w_q = self.w_q[:, query]
        # Compute query probability by summing the probability of all the worlds where the query is true
        query_prob = torch.sum(w_q * worlds_prob, dim=1, keepdim=True)
        return query_prob

    def problog_inference(self, facts_probs, labels=None, query=None):
        """
        Performs ProbLog inference to retrieve the worlds probability distribution P(w) and the desired query probability.
        """
        if query == None:
            pass

        n_digits = self.n_facts // 2
        # Extract first and second digit probability
        prob_digit1, prob_digit2 = facts_probs[:, 0], facts_probs[:, 1]
        # Compute worlds probability P(w) (the two digits values are independent)
        Z_1 = prob_digit1[..., None]
        Z_2 = prob_digit2[:, None, :]
        probs = Z_1.multiply(Z_2)
        self.worlds_prob = probs.reshape(-1, n_digits * n_digits)
        # Compute query probability P(q)
        query_prob = torch.zeros(size=(len(probs), self.nr_classes), device=probs.device)

        for i in range(self.nr_classes): #range(torch.bincount(used_labels).size(0)):
            query = i
            query_prob[:,i] = self.compute_query(query, self.worlds_prob).view(-1)

        # add a small bias
        eps = 1e-7
        query_prob += eps * torch.ones_like(query_prob, device=self.device)
        query_prob = (query_prob.T / query_prob.sum(-1 ) ).T
        return query_prob, self.worlds_prob

    def update_semiring_weights(self, facts_probs):
        """
        Updates weights of graph semiring with the facts probability.
        Each term probability is indicated as 'p_PositionDigit', we use it to index the priors contained in facts_probs.

        Args:
            facts_probs (bs, 2, n_facts)
        """
        for term in self.weights_dict:
            str_term = str(term)
            i = int(str_term[-2]) - 1
            j = int(str_term[-1])
            self.weights_dict[term] = facts_probs[:, i, j]

        self.semiring = GraphSemiring(facts_probs.shape[0], self.device)
        self.semiring.set_weights(self.weights_dict)

    def extract_query_probability(self, res):
        """Extracts P(q) contained in the dictionary 'res' resulting from ProbLog model evaluation."""
        res_keys = list(res.keys())
        return res[res_keys[-1]][..., None]

    def extract_worlds_probability(self, res):
        """Extracts P(w) contained in the dictionary 'res' resulting from ProbLog model evaluation."""
        n_digits = self.mlp.n_facts // 2
        digits = Term('digits')
        probabilities = []
        for j in range(n_digits):
            for k in range(n_digits):
                term = digits(Constant(j), Constant(k))
                probabilities.append(res[term])
        # Clamp probabilities to avoid nan
        probabilities = torch.stack(probabilities, dim=1)
        eps = 1e-7
        probabilities = probabilities + eps
        with torch.no_grad():
            P = probabilities.sum()
        return probabilities / P

    def get_params(self) -> torch.Tensor:
        """
        Returns all the parameters concatenated in a single tensor.
        :return: parameters tensor (??)
        """
        return self.encoder.get_params()

    def get_grads(self) -> torch.Tensor:
        """
        Returns all the gradients concatenated in a single tensor.
        :return: gradients tensor (??)
        """
        return self.encoder.get_grads()

    def get_grads_list(self):
        """
        Returns a list containing the gradients (a tensor for each layer).
        :return: gradients list
        """
        return self.encoder.get_grads_list()

    def set_params(self, new_params: torch.Tensor) -> None:
        """
        Sets the parameters to a given value.
        :param new_params: concatenated values to be set (??)
        """
        self.encoder.set_params(new_params)