# GPT-xs
GPT-xs is an Autoregresive text geneation model that replicate the GPT2 model created by OpenAI, it replicates the xs (very small) version of 124M parameters, this project
is inspired by the Andrew Karpahty video on YouTube, this model is not created for be used as other models in commerce is just done for personal practice.

**
The only parameters that differ from the GPT2 model architecture is the number of steps that has been decreased from 19073 to 10.000 and the context legnth also decreased from 
1024 to 512, the steps has been decreased because otherwise train if would have been much more expensive and i don't wanted and the context length has been decreased because 
the gpus that I used could not support a context length of 1024 for memory space issues
