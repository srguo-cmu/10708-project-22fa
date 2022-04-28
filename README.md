# 10708-project-22fa
This repo contains the scripts our team used for this project. It contains the training scripts of two models: VAE and GAN.

To run the models, you need to download the [Piano](http://deepyeti.ucsd.edu/cdonahue/wavegan/data/mancini_piano.tar.gz) dataset.

After the data is downloaded, run ``` python3 GAN_utils.py``` to sample the audios to chunks. The default chunk length is 4s. You can adjust it by changing the paramter in ```create_chunks(chunk_len=4)```


To run the GAN model: ()

``` python3 GANs.py```

The VAE model was implemented in jupyter notebook and one can run the VAE following the scripts in the notebook.
