# Stochastic_BB
Python implementation of SVRG-BB and SGD-BB algorithms of the following paper:

>["Barzilai-Borwein Step Size for Stochastic Gradient Descent". Conghui Tan, Shiqian Ma, Yu-Hong Dai, Yuqiu Qian. _NIPS 2016_.](http://papers.nips.cc/paper/6286-barzilai-borwein-step-size-for-stochastic-gradient-descent)

Please cite this paper if you use this code in your published research project.

## Files

- stochastic_bb.py: the source code for SVRG-BB and SGD-BB algorithms
- example.py: an example showing how to use these two algorithms


## Requirements

The code can be run with Python 3, with [numpy](http://www.numpy.org/) package installed. 

Theoretically speaking, this code should be compatible with Python 2. However, some users report numercial issues when running this code on Python 2.

But to run the example, [scipy](https://www.scipy.org/) and [theano](http://deeplearning.net/software/theano/) are also needed.

## License

MIT
