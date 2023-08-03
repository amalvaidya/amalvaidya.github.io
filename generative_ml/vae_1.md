---
title: "1. VAEs: Variational autoencoders"
---

# Starting simple: Autoencoders

Autoencoders are a good place to start since they'll demonstrate why the idea of being able to sample from a distribution is important. The way they work is also pretty intuitive. Autoencoders learn low dimensional representations by passing data through a "bottleneck" and try to reconstruct it at the other end. 


## What is an autoencoder?

You can break down an autoencoder network into two parts, the encoder and decoder. 
It works by passing data through the encoder, which takes the input and squeezes it down to vector that is smaller in size. This is the low dimensional bottleneck. This vector is then input to the decoder which tries to reconstruct the input image. The model is trained by minimizing the difference (more on this later) between the input image and reconstructed image. In order for this to work The output of the encoder needs to contain enough information about the input data so that the decoder network can recreate it.

![Autoencoder structure](images/autoencoder_structure.png)

The encoder output can be considered a low(er) dimensional vector representation of the input data. They're often called latent or hidden vectors since they're supposed to contain the latent attributes of the data. For our purposes though the most interesting part is the decoder. If you train an autoencoder on a collection of images then you should be able to generate new images by passing simple, low dimensional data into the decoder input!

## Defining a dataset

For this example we'll try and generate some faces from the famous [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset. The dataset contains more than 200 thousand images of celebrities. The images have also been cropped so that faces are centred.

::: {#fig-celeba layout-ncol=3}
![](images/celebA_val_set_23.png){}

![](images/celebA_val_set_123.png){}

![](images/celebA_val_set_453.png){}

![](images/celebA_val_set_623.png){}

![](images/celebA_val_set_645.png){}

![](images/celebA_val_set_2645.png){}

Examples from CelebA validation set
:::

The dataset also comes with a set of attributes labels for each image for things like "Blond Hair" and "Eyeglasses". These will be useful when we want to explore the encoded representations of the images that we generate. 


## Building and training the autoencoder

## Exploring the latent space

## Generating images!


# VAEs to the rescue

some words 

### The reparametrisation trick

## Adding the V in VAE





Intuitively 


> I remember asking a researcher at my old job exactly why VAEs made the assumption that elements in the latent space were normally distributed and the response was "that's just the assumption of the model. At the time it did not help.


#### References


