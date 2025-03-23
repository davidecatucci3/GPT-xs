# GPT-xs
![plot](plot.png)
GPT-xs is an autoregressive text generation model designed to replicate the GPT-2 architecture originally developed by OpenAI. This implementation specifically recreates the "xs" (extra small) variant, featuring 124 million parameters. The project draws inspiration from a YouTube video by Andrej Karpathy and was undertaken as a personal exercise for educational purposes. Unlike commercially deployed models, GPT-xs is not intended for production use but serves as a personal exercise to improve my skills in this field.


**
The only parameters that differ from the GPT2 model architecture is the number of steps that has been decreased from 19073 to 10.000 and the context legnth also decreased from 
1024 to 512, the steps has been decreased because otherwise train if would have been much more expensive and i don't wanted and the context length has been decreased because 
the gpus that I used could not support a context length of 1024 for memory space issues
