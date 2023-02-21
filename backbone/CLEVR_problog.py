import torch
from problog.logic import Constant, Term
from torch import nn
from backbone.utils.encoders import PairCNNEncoder
from backbone.utils.deepproblog_modules import GraphSemiring
from backbone.utils.deepproblog_modules import DeepProblogModel
from backbone.utils.utils_problog import build_worlds_queries_matrix

class CLEVR_DeepProblog(DeepProblogModel):
    def __init__(self, encoder= None,
                 model_dict=None,
                 n_facts=40, nr_classes=2, dropout=0.5,
                 is_train=True, device="cuda"):
        encoder=PairCNNEncoder(hidden_channels=64, latent_dim=40,dropout=0.5,img_channels=3) # PairMLPEncoder(28*28)
        model_dict= None
        super(CLEVR_DeepProblog, self).__init__(encoder=encoder, model_dict=model_dict,
                                                n_facts=n_facts, nr_classes=nr_classes, dropout=dropout,
                                                is_train=is_train, device=device)


        self.n_facts=n_facts
        self.w_s, self.w_c  = build_worlds_queries_matrix(2, 10,"CLEVR")
        self.w_s = self.w_s.to(device)
        self.w_c = self.w_c.to(device)

        # save dimension of concepts shape and colors
        self.dim_s = 10
        self.dim_c = 10

    def normalize_concepts(self, objs):
        """Computes the probability for each ProbLog fact given the latent vector z"""
        # Extract probs for each digit
        # Add stochasticity on prediction
        new_objs=[]
        for obj in objs:
            obj += 0.1 * torch.randn_like(obj, device=obj.device)
            # Sotfmax on concepts (the shapes and colors values are mutually exclusive)
            shape,color=torch.split(obj,self.dim_s,1)
            shape = nn.Softmax(dim=1)(shape)
            color = nn.Softmax(dim=1)(color)
            # Clamp digits_probs to avoid ProbLog underflow
            eps = 1e-5
            obj = obj + eps
            with torch.no_grad():
                Z1 = torch.sum(shape, dim=-1, keepdim=True).clone()
                Z2 = torch.sum(color, dim=-1, keepdim=True).clone()
            # Normalization
            shape = shape  / Z1
            color = color / Z2
            
            new_obj=torch.cat((shape,color),dim=1)
            new_objs.append(new_obj)


        return torch.stack(new_objs, dim=1) # now they are both probs

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

    def compute_query(self, query, worlds_prob, concept='shape'):
        """Computes query probability given the worlds probability P(w)."""
        # Select the column of w_q matrix corresponding to the current query
        if concept == 'shape':
            w_s = self.w_s[:, query]
            # Compute query probability by summing the probability of all the worlds where the query is true
            query_prob = torch.sum(w_s * worlds_prob, dim=1, keepdim=True)
            return query_prob
        elif concept == 'color':
            w_c = self.w_c[:, query]
            # Compute query probability by summing the probability of all the worlds where the query is true
            query_prob = torch.sum(w_c * worlds_prob, dim=1, keepdim=True)
            return query_prob
        else:
            return ValueError('Invalid choice')

    def problog_inference(self, facts_probs, labels=None, query=None):
        """
        Performs ProbLog inference to retrieve the worlds probability distribution P(w) and the desired query probability.
        """
        if query == None:
            pass


        # Extract first and second object probability
        obj1, obj2 =torch.split(facts_probs,1,1)
        shape1,color1= torch.split(obj1,10,2)
        shape2,color2= torch.split(obj2,10,2)

        # Compute worlds probability P(w) for shapes (the two objects are independent)
        Z_1 = shape1.reshape(-1,10,1)
        Z_2 = shape2.reshape(-1,1,10)


        # Compute worlds probability P(w) for colors (the two objects are independent)
        Z_3 = color1.reshape(-1,10,1)
        Z_4 = color2.reshape(-1,1,10)

        s_probs = torch.mul(Z_1,Z_2)
        c_probs = torch.mul(Z_3,Z_4)

        self.s_worlds_prob = s_probs.reshape(-1, self.dim_s * self.dim_s)
        self.c_worlds_prob = c_probs.reshape(-1, self.dim_c * self.dim_c)

        # Compute query probability P(q)
        s_query_prob = torch.zeros(size=(len(s_probs), self.nr_classes), device=c_probs.device)
        c_query_prob = torch.zeros(size=(len(c_probs), self.nr_classes), device=c_probs.device)

        for i in range(self.nr_classes):
            query = i
            s_query_prob[:,i] = self.compute_query(query, self.s_worlds_prob, concept='shape').squeeze()
            c_query_prob[:,i] = self.compute_query(query, self.c_worlds_prob, concept='color').squeeze()

        query_prob = torch.cat([s_query_prob, c_query_prob], dim=-1)
        query_prob = torch.zeros([len(self.s_worlds_prob), 4], device=facts_probs.device)
        #query_prob=torch.matmul(s_query_prob[:,:,None],c_query_prob[:,None,:]).view(-1,4)
        query_prob[:,0] = s_query_prob[:,0] * c_query_prob[:,0]
        query_prob[:,1] = s_query_prob[:,1] * c_query_prob[:,0]
        query_prob[:,2] = s_query_prob[:,0] * c_query_prob[:,1]
        query_prob[:,3] = s_query_prob[:,1] * c_query_prob[:,1]

        self.worlds_prob = torch.cat([self.s_worlds_prob, self.c_worlds_prob], dim=-1).clone()

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


    def forward(self, x):
        # Image encoding
        z = self.encoder(x)
        # normalize concept preditions
        self.facts_probs = self.normalize_concepts(z)
        # Problog inference to compute worlds and query probability distributions
        self.query_prob, self.worlds_prob = self.problog_inference(self.facts_probs)
        self.facts_probs=self.facts_probs.view(-1,40)
        return self.query_prob,self.facts_probs


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