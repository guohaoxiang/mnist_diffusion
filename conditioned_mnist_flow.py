import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, UNet2DModel, DDIMScheduler, FlowMatchEulerDiscreteScheduler
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from flowmatch import *

exp_name = 'flow_ori_net_bs1024_ep5'

# batch_size = 128
batch_size = 1024
n_epochs = 5
# n_epochs = 1
steps = 50
learning_rate = 1e-3

#craftsman version
noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)

denoise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps= 1000)



device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Load the dataset
dataset = torchvision.datasets.MNIST(root="mnist/", train=True, download=True, transform=torchvision.transforms.ToTensor())

# Feed it into a dataloader (batch size 8 here just for demo)
train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# View some examples
x, y = next(iter(train_dataloader))
print('Input shape:', x.shape)
print('Labels:', y)

plt.imsave(exp_name + '_1.png', torchvision.utils.make_grid(x)[0], cmap='Greys')

class ClassConditionedUnet(nn.Module):
  def __init__(self, num_classes=10, class_emb_size=4):
    super().__init__()
    
    # The embedding layer will map the class label to a vector of size class_emb_size
    self.class_emb = nn.Embedding(num_classes, class_emb_size)

    # Self.model is an unconditional UNet with extra input channels to accept the conditioning information (the class embedding)
    self.model = UNet2DModel(
        sample_size=28,           # the target image resolution
        in_channels=1 + class_emb_size, # Additional input channels for class cond.
        out_channels=1,           # the number of output channels
        layers_per_block=2,       # how many ResNet layers to use per UNet block
        block_out_channels=(32, 64, 64), 
        down_block_types=( 
            "DownBlock2D",        # a regular ResNet downsampling block
            "AttnDownBlock2D",    # a ResNet downsampling block with spatial self-attention
            "AttnDownBlock2D",
        ), 
        up_block_types=(
            "AttnUpBlock2D", 
            "AttnUpBlock2D",      # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",          # a regular ResNet upsampling block
          ),
    )

  # Our forward method now takes the class labels as an additional argument
  def forward(self, x, t, class_labels):
    # Shape of x:
    bs, ch, w, h = x.shape
    
    # import pdb; pdb.set_trace()
    # class conditioning in right shape to add as additional input channels
    class_cond = self.class_emb(class_labels) # Map to embedding dimension
    class_cond = class_cond.view(bs, class_cond.shape[1], 1, 1).expand(bs, class_cond.shape[1], w, h)
    # x is shape (bs, 1, 28, 28) and class_cond is now (bs, 4, 28, 28)

    # Net input is now x and class cond concatenated together along dimension 1
    net_input = torch.cat((x, class_cond), 1) # (bs, 5, 28, 28)

    # import pdb; pdb.set_trace()

    # Feed this to the UNet alongside the timestep and return the prediction
    return self.model(net_input, t).sample # (bs, 1, 28, 28)


# # ori scheduler
# noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
# denoise_scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')



#@markdown Training loop (10 Epochs):

# Redefining the dataloader to set the batch size higher than the demo of 8
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# train_dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)


# How many runs through the data should we do?

# Our network 
net = ClassConditionedUnet().to(device)

# Our loss function
loss_fn = nn.MSELoss()

# The optimizer
opt = torch.optim.Adam(net.parameters(), lr=learning_rate) 

# Keeping a record of the losses for later viewing
losses = []

#pre defined parames
weighting_scheme = 'logit_normal'
logit_mean = 0.0
logit_std = 1.0
mode_scale = 1.29

# The training loop
for epoch in range(n_epochs):
    for x, y in tqdm(train_dataloader):
        
        # Get some data and prepare the corrupted version
        x = x.to(device) * 2 - 1 # Data on the GPU (mapped to (-1, 1))
        y = y.to(device)
        noise = torch.randn_like(x)
        bsz = x.shape[0]
        u = compute_density_for_timestep_sampling(
            weighting_scheme=weighting_scheme,
            batch_size=bsz,
            logit_mean=logit_mean,
            logit_std=logit_std,
            mode_scale=mode_scale,
        )
        indices = (u * noise_scheduler.config.num_train_timesteps).long()
        timesteps = noise_scheduler.timesteps[indices].to(device)

        sigmas = get_sigmas(noise_scheduler, timesteps, n_dim=x.ndim, dtype=x.dtype, device = x.device)

        noisy_model_input = sigmas * noise + (1.0 - sigmas) * x

        model_pred = net(noisy_model_input, timesteps, y)

        model_pred = model_pred * (-sigmas) + noisy_model_input

        weighting = compute_loss_weighting_for_sd3(weighting_scheme=weighting_scheme, sigmas=sigmas)
        target = x

        loss = torch.mean(
            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            1,
        )
        loss = loss.mean()


        # # timesteps = torch.randint(0, 999, (x.shape[0],)).long().to(device)
        # noisy_x = noise_scheduler.add_noise(x, noise, timesteps)

        # # Get the model prediction
        # pred = net(noisy_x, timesteps, y) # Note that we pass in the labels y

        # # Calculate the loss
        # loss = loss_fn(pred, noise) # How close is the output to the noise

        # Backprop and update the params:
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Store the loss for later
        losses.append(loss.item())

    # Print out the average of the last 100 loss values to get an idea of progress:
    avg_loss = sum(losses[-100:])/100
    print(f'Finished epoch {epoch}. Average of the last 100 loss values: {avg_loss:05f}')

# View the loss curve
plt.plot(losses)
plt.savefig(exp_name + '_losses.png')

print('training finished, begin sampling')
#@markdown Sampling some different digits:

# Prepare random x to start from, plus some desired labels y
x = torch.randn(80, 1, 28, 28).to(device)
y = torch.tensor([[i]*8 for i in range(10)]).flatten().to(device)


timesteps, num_inference_steps = retrieve_timesteps(denoise_scheduler, steps, device, None)

for i, t in enumerate(timesteps):
    # expand the latents if we are doing classifier free guidance
    latent_model_input = x
    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
    timestep = t.expand(latent_model_input.shape[0])
    noise_pred = net(latent_model_input, timestep, y)

    # # perform guidance
    # if do_classifier_free_guidance:
    #     noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    #     noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # compute the previous noisy sample x_t -> x_t-1
    # latents_dtype = latents.dtype
    x = denoise_scheduler.step(noise_pred, t, x, return_dict=False)[0]

# Show the results
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
plt.imsave(exp_name + '_sampled.png', torchvision.utils.make_grid(x.detach().cpu().clip(-1, 1), nrow=8)[0], cmap='Greys')


#--------------------------------------

# # import pdb; pdb.set_trace()
# denoise_scheduler.set_timesteps(steps)
# timesteps = denoise_scheduler.timesteps.to(device)
# for i, t in enumerate(tqdm(timesteps, desc="DDIM Sampling:", leave=False)):
#     # Get model pred
#     with torch.no_grad():
#         residual = net(x, t, y)  # Again, note that we pass in our labels y

#     # Update sample with step
#     x = denoise_scheduler.step(residual, t, x).prev_sample


# # Show the results
# fig, ax = plt.subplots(1, 1, figsize=(12, 12))
# plt.imsave(exp_name + '_sampled.png', torchvision.utils.make_grid(x.detach().cpu().clip(-1, 1), nrow=8)[0], cmap='Greys')