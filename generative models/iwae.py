from torch import nn
import torch

class BasicBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BasicBlock, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.transform = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                       nn.Tanh(),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.Tanh())
        self.fc_mu = nn.Linear(hidden_dim, output_dim)
        self.fc_logsigma = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out = self.transform(x) # [num_samples, B, hidden_dim]
        mu = self.fc_mu(out) # [num_samples, B, dim_h1]
        logsigma = self.fc_logsigma(out) # [num_samples, B, dim_h1]
        sigma = torch.exp(logsigma) # [num_samples, B, dim_h1]
        return mu, sigma
    
class IWAE_1(nn.Module):
    def __init__(self, dim_h1, dim_image_vars):
        super(IWAE_1, self).__init__()
        self.dim_h1 = dim_h1
        self.dim_image_vars = dim_image_vars

        ## encoder
        self.encoder_h1 = BasicBlock(dim_image_vars, 200, dim_h1)
        
        ## decoder
        self.decoder_x =  nn.Sequential(nn.Linear(dim_h1, 200),
                                        nn.Tanh(),
                                        nn.Linear(200, 200),
                                        nn.Tanh(),  
                                        nn.Linear(200, dim_image_vars),
                                        nn.Sigmoid())
        
    def encoder(self, x):
        mu_h1, sigma_h1 = self.encoder_h1(x) # [num_samples, B, dim_h1], [num_samples, B, dim_h1]
        eps = sigma_h1.data.new(sigma_h1.size()).normal_() # [num_samples, B, dim_h1]
        h1 = mu_h1 + sigma_h1 * eps # [num_samples, B, dim_h1]
        return h1, mu_h1, sigma_h1, eps # h1, eps are random, mu_h1 and sigma_h1 are repeted num_samples times
    
    def decoder(self, h1):
        p = self.decoder_x(h1)
        return p
    
    def forward(self, x):
        h1, mu_h1, sigma_h1, eps = self.encoder(x)
        p = self.decoder(h1)
        return (h1, mu_h1, sigma_h1, eps), (p)

    def train_loss(self, inputs):
        h1, mu_h1, sigma_h1, eps = self.encoder(inputs)
        #log_Qh1Gx = torch.sum(-0.5*((h1-mu_h1)/sigma_h1)**2 - torch.log(sigma_h1), -1)
        log_Qh1Gx = torch.sum(-0.5*(eps)**2 - torch.log(sigma_h1), -1) # same as above line, Q(h) Given G(x)
        # => [num_samples, B]
        
        p = self.decoder(h1) # [num_samples, B, 784]
        log_Ph1 = torch.sum(-0.5*h1**2, -1) # [num_samples, B]
        log_PxGh1 = torch.sum(inputs*torch.log(p) + (1-inputs)*torch.log(1-p), -1) # [num_samples, B]

        log_weight = log_Ph1 + log_PxGh1 - log_Qh1Gx # [num_samples, B]
        log_weight = log_weight - torch.max(log_weight, 0)[0] # [num_samples, B], divide by max_weight from num_samples
        weight = torch.exp(log_weight) # [num_samples, B]
        weight = weight / torch.sum(weight, 0) # [num_samples, B]
        loss = -torch.mean(torch.sum(weight.detach() * (log_Ph1 + log_PxGh1 - log_Qh1Gx), 0))
        
        bce = (-log_PxGh1).mean()
        kld = (log_Qh1Gx - log_Ph1).mean()
        return loss, bce, kld

    def test_loss(self, inputs):
        h1, mu_h1, sigma_h1, eps = self.encoder(inputs)
        #log_Qh1Gx = torch.sum(-0.5*((h1-mu_h1)/sigma_h1)**2 - torch.log(sigma_h1), -1)
        log_Qh1Gx = torch.sum(-0.5*(eps)**2 - torch.log(sigma_h1), -1) # log(q(h1|x)), [num_samples, B]
        
        p = self.decoder(h1)
        log_Ph1 = torch.sum(-0.5*h1**2, -1) # log(p(h1)), [num_samples, B] 
        log_PxGh1 = torch.sum(inputs*torch.log(p) + (1-inputs)*torch.log(1-p), -1) # log(p(x|h1)), binary cross entropy, [num_samples, B]

        log_weight = log_Ph1 + log_PxGh1 - log_Qh1Gx
        weight = torch.exp(log_weight) # p(x|h1) * p(h1) / p(h1|x), [num_samples, B] 
        loss = -torch.mean(torch.log(torch.mean(weight, 0)))# [num_samples, B] -> [B] -> scalar
        return loss