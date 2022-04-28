# 10708-project-22fa
This repo contains the scripts our team used for this project. It contains the training scripts of two models: VAE and GAN.



***Dataset***

To run the models, you need to download the [Piano](http://deepyeti.ucsd.edu/cdonahue/wavegan/data/mancini_piano.tar.gz) dataset. After the data is downloaded, run ```$ python3 GAN_utils.py``` to sample the audios to chunks. The default chunk length is 4s. You can adjust it by changing the paramter in ```create_chunks(chunk_len=4)```

***GAN***

To run the GAN model, first specify the path of your data in ```Piano_DS()``` when creating the dataset(TO DO WHEN scripts is finalized). Then run
```
$ python3 GANs.py --batch_size BATCH --lr LR --data_dir /path/to/data --G_d GD --D_d DD --z_size Z
```

***VAE***

The VAE model was implemented in jupyter notebook and one can run the VAE by following the scripts in the notebook.
