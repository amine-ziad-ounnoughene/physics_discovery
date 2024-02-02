import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SciNet(nn.Module):
	def __init__(self, input_dim, output_dim, latent_dim, layer_dim):
		"""Initialize SciNet Model.
		
		Params
		======
			input_dim (int): number of inputs
			output_dim (int): number of outputs
			latent_dim (int): number of latent neurons
			Layer_dim (int): number of neurons in hidden layers
		"""
		super(SciNet, self).__init__()
		self.latent_dim = latent_dim
		self.enc1 = nn.Linear(input_dim, 500)
		self.enc2 = nn.Linear(500, 300)
		self.enc3 = nn.Linear(300, 100)
		self.latent = nn.Linear(100, latent_dim*2)
		self.dec1 = nn.Linear(latent_dim+1, 100)
		self.dec2 = nn.Linear(100,100)
		self.out = nn.Linear(100, output_dim)
		
	def encoder(self, x):
		z = F.relu(self.enc1(x))
		z = F.relu(self.enc2(z))
		z = F.relu(self.enc3(z))
		z = self.latent(z)
		self.mu = z[:, 0:self.latent_dim]
		self.sigma = z[:, self.latent_dim:]
		# self.sigma = torch.exp(self.log_sigma)        

		# Use reparametrization trick to sample from gaussian
		eps = torch.randn(x.size(0), self.latent_dim).to(device)
		z_sample = self.mu + self.sigma * eps        

		# Compute KL loss
		self.kl_loss = kl_divergence(self.mu, self.sigma, dim=self.latent_dim)
		return z_sample
	
	def decoder(self, z):
		x = F.relu(self.dec1(z))
		x = F.relu(self.dec2(x))    
		return self.out(x)

	def forward(self, obs):
		q = obs[:,-1].reshape(obs.size(0),1)
		# print('question: ', q)
		obs = obs[:,0:-1]
		# print('obs: ', obs)
		self.latent_r = self.encoder(obs) 
		# print('latent_r: ', self.latent_r)
		dec_input = torch.cat( (q, self.latent_r), 1)

		return self.decoder(dec_input)


def kl_divergence(means, log_sigma, dim, target_sigma=0.1):
	"""
	Computes Kullback–Leibler divergence for arrays of mean and log(sigma)
	"""
	target_sigma = torch.Tensor([target_sigma]).to(device)
	return 1 / 2. * torch.mean(torch.mean(1 / target_sigma**2 * means**2 +
			torch.exp(2 * log_sigma) / target_sigma**2 - 2 * log_sigma + 2 * torch.log(target_sigma), dim=1) - dim)



   