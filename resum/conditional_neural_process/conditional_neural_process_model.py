import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class DeterministicEncoder(nn.Module):
    def __init__(self, output_sizes):
        super(DeterministicEncoder, self).__init__()
        self.linears = nn.ModuleList()
        for i in range(len(output_sizes) - 1):
            self.linears.append(nn.Linear(output_sizes[i], output_sizes[i + 1]))

    def forward(self, context_x, context_y):
        """Encodes the inputs into one representation.

        Args:
        context_x: Tensor of size of batches x observations x m_ch. For this 1D regression
          task this corresponds to the x-values.
        context_y: Tensor of size bs x observations x d_ch. For this 1D regression
          task this corresponds to the y-values.

        Returns:
            representation: The encoded representation averaged over all context 
            points.
        """

        # Concatenate x and y along the filter axes
        encoder_input = torch.cat((context_x, context_y), dim=-1)

        # Get the shapes of the input and reshape to parallelise across observations
        batch_size, num_context_points, _ = encoder_input.shape
        hidden = encoder_input.view(batch_size * num_context_points, -1)
        
        # Pass through MLP
        for i, linear in enumerate(self.linears[:-1]):
            hidden = torch.relu(linear(hidden))
        # Last layer without a ReLu
        hidden = self.linears[-1](hidden)
        # Bring back into original shape (# Flatten the output feature map into a 1D feature vector)
        hidden = hidden.view(batch_size, num_context_points, -1)

        # Aggregator: take the mean over all points
        representation = hidden.mean(dim=1)
        return representation

def sigmoid_expectation(mu, sigma):
    # Bound the variance
    sigma = 0.1 + 0.9*torch.nn.functional.softplus(sigma)
    #sigma = 0.01 + 0.99*torch.nn.functional.softplus(sigma)
    
    y = torch.from_numpy(np.sqrt(1+3/np.pi**2*sigma.detach().numpy()**2))
    # Bound the divisor to > 0
    tmp0 = torch.where(y==0.,1e-4,0.)
    y=torch.add(y,tmp0)
    
    expectation = torch.sigmoid(mu/y) 
    var = expectation * (1-expectation) * (1-(1/y))
    #var = 0.01 + 0.99*torch.nn.functional.softplus(var)
    #tmp = torch.where(var==0.,1.e-4,0.)
    #var = torch.add(expectation, tmp)
    
    return expectation, var

class DeterministicDecoder(nn.Module):
    def __init__(self, output_sizes):
        """CNP decoder.
        Args:
            output_sizes: An iterable containing the output sizes of the decoder MLP.
        """
        super(DeterministicDecoder, self).__init__()
        self.linears = nn.ModuleList()
        for i in range(len(output_sizes) - 1):
            self.linears.append(nn.Linear(output_sizes[i], output_sizes[i + 1]))

    def forward(self, representation, target_x):
        """Decodes the individual targets.

        Args:
            representation: The encoded representation of the context
            target_x: The x locations for the target query

        Returns:
            dist: A multivariate Gaussian over the target points.
            mu: The mean of the multivariate Gaussian.
            sigma: The standard deviation of the multivariate Gaussian.   
        """

        # Get the shapes of the input and reshape to parallelise across observations
        batch_size, num_total_points, _ = target_x.shape
        representation = representation.unsqueeze(1).repeat([1, num_total_points, 1])

        # Concatenate the representation and the target_x
        input = torch.cat((representation, target_x), dim=-1)
        hidden = input.view(batch_size * num_total_points, -1)

        # Pass through MLP
        for i, linear in enumerate(self.linears[:-1]):
            hidden = torch.relu(linear(hidden))
        # Last layer without a ReLu
        hidden = self.linears[-1](hidden)

        # Bring back into original shape
        hidden = hidden.view(batch_size, num_total_points, -1)

        # Get the mean an the variance
        mu, sigma = torch.split(hidden, 1, dim=-1)
        
        # Map mu to a value between 0 and 1 and get the expectation and variance
        mu, sigma = sigmoid_expectation(mu, sigma)

        # Get the distribution
        # Ensure scale is strictly positive before passing to Normal distribution
        #sigma = torch.clamp(sigma, min=1e-4)  # OR
        sigma = F.softplus(sigma) + 1e-6

        dist = torch.distributions.normal.Normal(loc=mu, scale=sigma)
        return dist, mu, sigma

class DeterministicModel(nn.Module):
    def __init__(self, encoder_sizes, decoder_sizes):
        super(DeterministicModel, self).__init__()
        """Initialises the model.

        Args:
            encoder_output_sizes: An iterable containing the sizes of hidden layers of
                the encoder. The last one is the size of the representation r.
            decoder_output_sizes: An iterable containing the sizes of hidden layers of
                the decoder. The last element should correspond to the dimension of
                the y * 2 (it encodes both mean and variance concatenated)
        """
        self._encoder = DeterministicEncoder(encoder_sizes)
        self._decoder = DeterministicDecoder(decoder_sizes)

    def forward(self, query, target_y=None):
        """Returns the predicted mean and variance at the target points.

        Args:
            query: Array containing ((context_x, context_y), target_x) where:
                context_x: Array of shape batch_size x num_context x 1 contains the 
                    x values of the context points.
                context_y: Array of shape batch_size x num_context x 1 contains the 
                    y values of the context points.
                target_x: Array of shape batch_size x num_target x 1 contains the
                    x values of the target points.
            target_y: The ground truth y values of the target y. An array of 
                shape batchsize x num_targets x 1.

        Returns:
            log_p: The log_probability of the target_y given the predicted
            distribution.
            mu: The mean of the predicted distribution.
            sigma: The variance of the predicted distribution.
        """

        (context_x, context_y), target_x = query
        # Pass query through the encoder and the decoder

        representation = self._encoder(context_x, context_y)
        dist, mu, sigma = self._decoder(representation, target_x)
        
        # If we want to calculate the log_prob for training we will make use of the
        # target_y. At test time the target_y is not available so we return None
        log_p = None if target_y is None else dist.log_prob(target_y)
        return log_p, mu, sigma

