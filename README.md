# 10708-project-22fa
This repo contains the scripts our team used for this project. It contains the training scripts of two models: VAE and GAN.



***Dataset***

To run the models, you need to download the [Piano](http://deepyeti.ucsd.edu/cdonahue/wavegan/data/mancini_piano.tar.gz) dataset. Make sure the individual .wav files are put in a directory named "piano". After the data is downloaded, run ```$ python3 GAN_utils.py``` to sample the audios to chunks(you need to change the path of the data). The default chunk length is 4s. You can adjust it by changing the paramter in ```create_chunks(chunk_len=4)```

***GAN***

These instructions assume you have already downloaded and chunked the data.

To train a model with either WaveGAN or CMD WaveGAN discriminator/generator, run the command

```
$ python3 train.py
```

The possible command-line arguments are as follows:

```
--epochs", type=int, default=500: number of epochs
--lr", type=float, default=0.0001: learning rate
--batch_size", type=int, default=64: batch_size
--z_size", type=int, default=400: random vector size in generator
--G_d", type=int, default=64: generator dimensionality parameter
--D_d", type=int, default=64: discriminator dimensionality parameter
--G_type, type=str, default="old": generator type, old (vanilla WaveGAN) vs new (CMD WaveGAN)
--D_type, type=str, default="old": discriminator type, old (vanilla WaveGAN) vs new (CMD WaveGAN)
--k, type=int, default=5: number of discriminator sub-epochs per generator epoch
--l, type=float, default=10: lambda parameter, weighting on gradient penalty
--subsample_ratio, type=float, default=1: subsampling ratio of data
--inject_noise_var, type=float, default=0: variance of injected gaussian noise to data
--save_results", type=bool, default=True: if we wish to save results to a folder. If this is true, a directory named 'GAN_results_piano' needs to exist.
```


***VAE***

The VAE model was implemented in jupyter notebook and one can run the VAE by following the scripts in the notebook.


***Robustness Experiment***

To run the robustness analysis, follow the example in the script ``` robustness_eval_fns.py```

