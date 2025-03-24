# GPT-xs

<p align="center">
  <img src="meme.gif" alt="plot">
  <br>
  I was searching an image to put and this just makes me laugh a lot, so i put it idkw
</p>

GPT-xs is an autoregressive text generation model designed to replicate the GPT-2 architecture originally developed by OpenAI. This implementation specifically recreates the "xs" (extra small) variant, featuring 124 million parameters. The project draws inspiration from a YouTube video by Andrej Karpathy and was undertaken as a personal exercise for educational purposes. Unlike commercially deployed models, GPT-xs is not intended for production use but serves as a personal exercise to improve my skills in this field.

## Files
- data_loader.py: Implements functionality to efficiently load batches of data from preprocessed shards, preparing them for input into the model during training or inference
- dataset.py: Handles the retrieval of the dataset from Hugging Face, processes it into tokenized shards, and prepares the data for use in model training
- model.py: Defines the architecture and structure of the machine learning model, specifying layers, parameters, and configurations
- train.py: Orchestrates the training process, including loading the model, optimizing parameter and managing the training loop for effective learning
- eval.py: Used to use the model after training
- plot.py: Plot the data that have been collected during the training process
- hellaswag.py: Calculate the accuracy of the model using the HellaSwag dataset


**
The only parameters that differ from the GPT2 model architecture is the number of steps that has been decreased from 19073 to 10.000 and the context legnth also decreased from 
1024 to 512, the steps has been decreased because otherwise train if would have been much more expensive and i don't wanted and the context length has been decreased because 
the gpus that I used could not support a context length of 1024 for memory space issues
