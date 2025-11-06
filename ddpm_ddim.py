import torch
import os
from step1_utils.models.unet import create_model
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import step1_utils.utils as utils
import argparse

class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.conf = None
        
        # hyperparameters for path & dataset
        self.parser.add_argument('--out_path', type=str, default='step1_results', help='results file directory')
        self.parser.add_argument('--dataset', type=str, default='ffhq', help='either choose ffhq or imagenet')
        
        # hyperparameters for sampling
        self.parser.add_argument('--total_instances', type=int, default=10, help='number of images you want to generate - 10 is ideal')
        self.parser.add_argument('--diff_timesteps', type=int, default=1000, help='Original number of steps from Ho et al. (2020) which is 1000 - do not change')
        self.parser.add_argument('--desired_timesteps', type=int, default=1000, help='How many steps do you want?')
        self.parser.add_argument('--eta', type=float, default=0.0, help='Should be between [0.0, 1.0]')
        self.parser.add_argument('--schedule', type=str, default="1000", help="regular/irregular schedule to use (jumps)")

    def parse(self, args=None):
        """Parse the configuration"""
        self.conf = self.parser.parse_args(args=args)
        return self.conf

class Sampler():
    def __init__(self):
        self.conf = Config().parse()
        scale = 1000 / self.conf.diff_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        self.betas = torch.linspace(beta_start, beta_end, self.conf.diff_timesteps, dtype=torch.float64)
        self.alpha_init()
       
    def alpha_init(self):
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = torch.cat((torch.tensor([1.0]), self.alphas_cumprod[:-1]))
    
    def recreate_alphas(self):
        use_timesteps = utils.space_timesteps(self.conf.diff_timesteps, self.conf.schedule) # Selects a subset of timesteps according to the given schedule
        self.timestep_map = []
        last_alpha_cumprod = 1.0
        
        #  Initialize an empty list to store the new beta values --> This is to collect the new betas corresponding to the chosen timesteps.
        new_betas = []

        for i, alpha_cumprod in enumerate(self.alphas_cumprod):
            # pass # Delete this
            # TODO: Check if the current timestep index 'i' is part of the selected timesteps (use_timesteps)
            if i in use_timesteps:    
    
                beta_new = 1.0 - (float(alpha_cumprod) / float(last_alpha_cumprod))
                beta_new = max(min(beta_new, 0.999), 1e-12)
                new_betas.append(beta_new)
            # TODO: Update 'last_alpha_cumprod' to the current 'alpha_cumprod'
                last_alpha_cumprod = float(alpha_cumprod)
           
                self.timestep_map.append(i)

        # TODO: Convert 'new_betas' into a PyTorch tensor and store it in 'self.betas'
        self.betas = torch.tensor(new_betas, dtype=torch.float64)
        # TODO: After updating betas, Recompute the related alpha terms to refresh alpha values
        # Hint: A helper function is already implemented in this hw3_step1_main.py file to refresh the alpha values
        # Understand which function does that and use it here.
        self.alpha_init()
        
        return torch.tensor(self.timestep_map)
    
    def get_variance(self, x, t):
        posterior_variance = (self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        posterior_log_variance_clipped = torch.log(torch.cat((posterior_variance[1].unsqueeze(0), posterior_variance[1:])))
        posterior_variance = (self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        posterior_log_variance_clipped = torch.log(torch.cat((posterior_variance[1].unsqueeze(0), posterior_variance[1:])))
        model_var_values = x
        min_log = posterior_log_variance_clipped
        max_log = torch.log(self.betas)
        min_log = utils.extract_and_expand(min_log, t, x)
        max_log = utils.extract_and_expand(max_log, t, x)
        frac = (model_var_values + 1.0) / 2.0
        model_log_variance = frac * max_log + (1-frac) * min_log
        return model_log_variance
    
    def predict_x0_hat(self, x_t, t, model_output):
        ############################
        # Implementation of the function predicting the clean denoised estimate x_{0|t}
        ############################
        Ct_arr = 1.0 / torch.sqrt(self.alphas_cumprod)
        Dt_arr = - torch.sqrt(1.0 - self.alphas_cumprod) / torch.sqrt(self.alphas_cumprod)

        Ct = utils.extract_and_expand(Ct_arr, t, x_t)
        Dt = utils.extract_and_expand(Dt_arr, t, x_t)

        x0_hat = Ct * x_t + Dt * model_output
        return x0_hat
    
    def sample_ddpm(self, score_model, x_t, model_t, t):
        with torch.no_grad():
            model_output = score_model(x_t, model_t)
        model_output, model_var_values = torch.split(model_output, x_t.shape[1], dim=1)
        model_log_variance = self.get_variance(model_var_values, t)
        
      
        # Implementing DDPM sampling 
        ############################
        At_arr = 1.0 / torch.sqrt(self.alphas)  # shape: [#steps]
        Bt_arr = - self.betas / (torch.sqrt(self.alphas) * torch.sqrt(1.0 - self.alphas_cumprod))

        At = utils.extract_and_expand(At_arr, t, x_t)
        Bt = utils.extract_and_expand(Bt_arr, t, x_t)

        mu = At * x_t + Bt * model_output

        # σ_t from predicted log-variance
        sigma = torch.exp(0.5 * model_log_variance)

        # z ~ N(0, I) if t > 0; else z = 0
        # (all elements in t are equal per batch)
        add_noise = (t > 0).view(-1, *([1] * (x_t.dim() - 1))).to(x_t.dtype)
        z = torch.randn_like(x_t) * add_noise

        sample = mu + sigma * z

        return sample

    def sample_ddim(self, score_model, x_t, model_t, t):
        with torch.no_grad():
            model_output = score_model(x_t, model_t)
        model_output, _ = torch.split(model_output, x_t.shape[1], dim=1)
        
        ############################
        # Implementing DDIM sampling 
        ############################
        eta = self.conf.eta

        a_bar_t      = utils.extract_and_expand(self.alphas_cumprod,      t, x_t)
        a_bar_prev   = utils.extract_and_expand(self.alphas_cumprod_prev, t, x_t)
        eps_hat      = model_output
        x0_hat       = self.predict_x0_hat(x_t, t, eps_hat)

        # σ_t per DDIM
        sigma_t = eta * torch.sqrt((1.0 - a_bar_prev) / (1.0 - a_bar_t)) * torch.sqrt(1.0 - a_bar_t / a_bar_prev)

        # Coefficient for ε̂
        eps_coeff = torch.sqrt(1.0 - a_bar_prev - sigma_t * sigma_t)

        # Mean (deterministic part)
        mu = torch.sqrt(a_bar_prev) * x0_hat + eps_coeff * eps_hat

        # Noise: z ~ N(0, I) if t > 0 else 0
        add_noise = (t > 0).view(-1, *([1] * (x_t.dim() - 1))).to(x_t.dtype)
        z = torch.randn_like(x_t) * add_noise

        sample = mu + sigma_t * z
        
        return sample
    
def main():
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    conf = Config().parse()
    
    algo = "DDPM"
    if algo == "DDIM":
        print('*' * 60 + f'\nSTARTED {algo} Sampling with eta = \"%.1f\" \n' %conf.eta)
    if algo == "DDPM":
        print('*' * 60 + f'\nSTARTED {algo} Sampling')
        
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Create and config model
    model_config = utils.load_yaml("step1_utils/models/" + conf.dataset + "_model_config.yaml")
    score_model = create_model(**model_config).to(device).eval()

    # Sampling
    for instance in range(conf.total_instances):
        sampler_operator = Sampler()
        x_t = utils.get_noise_x_t(instance, conf.total_instances, device)
        pbar = (list(range(conf.desired_timesteps))[::-1])
        
        if conf.desired_timesteps == 1000:
            time_map = torch.tensor(list(utils.space_timesteps(conf.diff_timesteps, "1000"))).to(device)
        else:
            time_map = sampler_operator.recreate_alphas().to(device)
        
        print(f"\n********* image {instance+1}/{conf.total_instances}: *********\n")
        for idx in tqdm(pbar):
            time = torch.tensor([idx] * x_t.shape[0], device=device)
            if algo == "DDPM":
                x_t_prev_bar = sampler_operator.sample_ddpm(score_model, x_t, time_map[time], time)
            if algo == "DDIM":
                x_t_prev_bar = sampler_operator.sample_ddim(score_model, x_t, time_map[time], time)
            x_t = x_t_prev_bar
        image_filename = f"image_{instance+1}.png"
        
        image_path = os.path.join(conf.out_path, image_filename)
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        plt.imsave(image_path, utils.clear_color(x_t))
    print('\nFINISHED Sampling!\n' + '*' * 60)

if __name__ == '__main__':
    main()
