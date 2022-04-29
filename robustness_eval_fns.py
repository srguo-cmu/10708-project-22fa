"""
Collection of functions and example usage for metrics about GAN generated
audio samples. Metrics include Generator loss, D_sample, and D_train
"""

import numpy as np
import torch

def NN_dists(refs, samples, curr_min=None, kth=0):
  
  mins = np.zeros(len(samples))
  for idx in range(len(samples)):
    dist = np.linalg.norm(refs - np.expand_dims(samples[idx, :], 0), axis=1) # (N,)
    min_dist = np.partition(dist, kth)[kth] # (1,)

    if curr_min is None:
      mins[idx] = min_dist
    else:
      mins[idx] = min(min_dist, curr_min[idx])

  return mins

def get_train_D(dl, samples):

  curr_min = None
  for i, batch in enumerate(dl):
    if i%5 == 0:
      print('Batch ' + str(i))
    refs = batch[0].squeeze().numpy()
    curr_min = NN_dists(refs, samples, curr_min)
  return np.mean(curr_min)

def get_sample_D(samples):

  dists = NN_dists(samples, samples, kth=1)
  return np.mean(dists)


"""
Usage:

# The Usage of these functions to produce metrics for evaluating the
# quality of generated samples is dependent on the path structure and 
# naming convention used by the user. Therefore, an example run as used
# for this project is included in comment form below.

collector = {}
collector_D = {}
collector_Dnoise = {}
collector_distTrain = {}
collector_distSamples = {}
combs = [(0.1, 0), (0.5, 0), (0.5, 0.01), (0.5, 0.1), (0.5, 0.02), (0.5, 0.05), (0.5, 0.15), (0.25, 0), (0.25, 0.01), (0.25, 0.1), (0.25, 0.02), (0.25, 0.05), (0.25, 0.15)]
for subsamp, noisevar in combs:
  ston_list = []
  tot_G_loss = 0
  Gnoise_loss = 0

  ## Load the data
  save_direc = os.path.join("trained_models", "4sec_Gd=64_Dd=64_e=750_lr=0.0001_b=64_z=100_k=5_subsamp="+str(subsamp)+"_noisevar="+str(noisevar))
  loss_dict, G, D = None, None, None
  with open(os.path.join(save_direc, "loss_trajecs_750.pkl"), "rb") as f:
    loss_dict = pickle.load(f)
  with open(os.path.join(save_direc, "G_750.pt"), "rb") as f:
    G = torch.load(f)
  with open(os.path.join(save_direc, "D_750.pt"), "rb") as f:
    D = torch.load(f)
  G = Generator(input_size=100, d=64)
  G.load_state_dict(torch.load( os.path.join(save_direc, "G_750.pt"), map_location=torch.device("cpu") ))
  D_noise = Discriminator(d=64, device=device)
  D_noise.load_state_dict(torch.load( os.path.join("trained_models", "4sec_Gd=64_Dd=64_e=750_lr=0.0001_b=64_z=100_k=5_subsamp=0.5_noisevar="+str(noisevar), "D_750.pt"), map_location=torch.device("cpu") ))
  D = Discriminator(d=64, device=device)
  print(D)
  D.load_state_dict(torch.load( os.path.join("trained_models", "full", "D_1500.pt"), map_location=torch.device("cpu") ))
  ##

  ## Make loss plots
  G_losses = loss_dict["G"]
  D_losses = loss_dict["D"]
  W_losses = loss_dict["W"]

  plt.figure()
  plt.plot(range(len(D_losses)), D_losses, label="D")
  plt.legend(loc="upper right")
  plt.title("Loss Trajectory, Discriminator, Piano")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.savefig(save_direc + "/piano_D_loss_trajecs.png")

  plt.figure()
  plt.plot(range(len(G_losses)), G_losses, label="G", c="orange")
  # plt.plot(range(len(D_losses)), D_losses, label="D")
  plt.legend(loc="upper right")
  plt.title("Loss Trajectory, Generator, Piano")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.savefig(save_direc + "/piano_G_loss_trajecs.png")

  plt.figure()
  plt.plot(range(len(W_losses)), W_losses, label="W", c="red")
  # plt.plot(range(len(D_losses)), D_losses, label="D")
  plt.legend(loc="upper right")
  plt.title("Loss Trajectory, Wasserstein, Piano")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.savefig(save_direc + "/piano_W_loss_trajecs.png")
  ##

  ## Generate samples and ston
  tot_ston = 0
  num_samples = 1000
  gen_samples = []
  avg_train_D = 0

  G=G.to(device)
  D=D.to(device)
  D_noise = D_noise.to(device)
  import soundfile as sf
  for i in range(num_samples):
    z= (torch.rand(size=(1,100))*2-1).to(device)
    gen_sound = G( z )
    y_pred_fake = D(gen_sound)
    noise_ypredfake = D_noise(gen_sound)
    gen_sound= gen_sound.detach().cpu().numpy()[0][0]

    if i < 10:
      print(gen_sound.shape)
      sound( gen_sound,rate=4096)
      plt.figure()
      plt.plot(range(len(gen_sound)),gen_sound)
      plt.savefig(save_direc + "/piano_sample{}_waveform.png".format(i))
      sf.write(save_direc + "/piano_sample{}.wav".format(i), gen_sound, 4096)

    tot_ston += s_to_n(gen_sound)
    ston_list.append(s_to_n(gen_sound))

    tot_G_loss -= y_pred_fake.detach().cpu()
    Gnoise_loss -= noise_ypredfake.detach().cpu()

    gen_samples.append(gen_sound)

  gen_samples = np.array(gen_samples)
  avg_sample_D = get_sample_D(gen_samples)
  print(avg_sample_D)
  avg_train_D = get_train_D(dataloader, gen_samples)
  print(avg_train_D)
  avg_train_D /= num_samples
  tot_ston /= num_samples
  tot_G_loss /= num_samples
  Gnoise_loss /= num_samples
  print(tot_ston)
  # collector[(subsamp, noisevar)] = tot_ston
  collector[(subsamp, noisevar)] = (np.mean(ston_list), 1.96*np.std(ston_list)/np.sqrt(1000))
  collector_D[(subsamp, noisevar)] = tot_G_loss
  collector_Dnoise[(subsamp, noisevar)] = Gnoise_loss
  collector_distTrain[(subsamp, noisevar)] = avg_train_D
  collector_distSamples[(subsamp, noisevar)] = avg_sample_D
  ##
"""