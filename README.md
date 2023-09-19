# AudioGenerator
# Overview
This is a repository that attempts to compare and constrast the efficacy of audio generation between the Variational Autoencoder and the U-Net architecture. The objective of the models is to reproduce sound that represents human-spoken digits, i.e., "one", "two", etc. Robustness is a secondary objective, e.g., if we change the input data slightly, will the models produce a similar sound? This is extensively detailed in the [audio_generation.pdf](https://github.com/jsturges28/AudioGenerator/blob/main/audio_generation.pdf) paper.
# Contents
This repo contains the following:
* Hundreds of voice recordings from the Free Spoken Digit Dataset (FSDD).
* **Varational Autoencoder**: Contains and encoder and decoder portion, with the reparameterization trick. [Source](https://ermongroup.github.io/cs228-notes/extras/vae/)
* **UNET**: A slighty modified UNET model that contains ~469k parameters. [Source](https://link.springer.com/content/pdf/10.1007/978-3-319-24574-4_28.pdf)
* **COMING SOON: UNET 3+**: A highly modified UNET 3+ model that contains ~4 million parameters. [Source](https://ieeexplore.ieee.org/stampPDF/getPDF.jsp?tp=&arnumber=9053405&ref=aHR0cHM6Ly9pZWVleHBsb3JlLmllZWUub3JnL2Fic3RyYWN0L2RvY3VtZW50LzkwNTM0MDU=&tag=1)

### TODO:
- [ ] Add U-NET 3+ support and complete U-NET 3+
- [ ] Format general filepath support for loading data
- [ ] Add Dice/F1 metrics
- [ ] Add additional callback support for IOU metrics
- [ ] Add argparse functionality for toggling analytics
