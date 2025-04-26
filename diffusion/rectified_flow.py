import torch
from tqdm import tqdm
class RectifiedFlow():
    def __init__(self, num_timesteps, noise_scale=1.0, init_type='gaussian', eps=1, sampling="log"):
        """
        eps: A `float` number. The smallest time step to sample from.
        """
        self.num_timesteps = num_timesteps
        self.T = 1000.
        self.noise_scale = noise_scale
        self.init_type = init_type
        self.eps = eps
        self.sampling = sampling

    def training_loss(self, model, x, model_kwargs):
        if self.sampling == "log":
            normal = torch.randn(x.shape[0], device=x.device)
            t = torch.sigmoid(normal) *(self.T-self.eps) + self.eps
        elif self.sampling == "uniform":
            t = torch.rand(x.shape[0], device=x.device)

        z0 = self.get_z0(x).to(x.device)

        t_expand = t.view(-1,1,1,1).repeat(1,x.shape[1],x.shape[2],x.shape[3]) / self.T

        pertubed_data = t_expand*x + (1-t_expand)*z0

        score = model(pertubed_data, t, **model_kwargs)

        target = x - z0 # direction of the flow

        losses = torch.square(score-target)
        return {"loss":torch.mean(losses)}

    def sample(self, model, z, model_kwargs, progress=True):
        dt = 1/float(self.num_timesteps)

        for i in tqdm(range(self.num_timesteps), disable=not progress):
            num_t = i/self.num_timesteps * (self.T-self.eps)+self.eps

            t = torch.ones(z.shape[0], device=z.device).reshape((z.shape[0],)) * num_t

            v = model(z, t, **model_kwargs)

            z = z.detach().clone() + dt*v
        
        return z
    def get_z0(self, batch, train=True):

        if self.init_type == 'gaussian':
            ### standard gaussian #+ 0.5
            return torch.randn(batch.shape)*self.noise_scale
        else:
            raise NotImplementedError("INITIALIZATION TYPE NOT IMPLEMENTED") 