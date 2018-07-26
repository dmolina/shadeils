This source code implement the winner of the Large-Scale Global Optimization
Competition organized in IEEE Congress of Evolutionary Computation 2018, 
http://www.tflsgo.org/special_sessions/cec2018.html

The implementation is done in Python 3, using numpy.

This source code is freely available under the General Public License (GPLv3).
However, if you use it in a research paper, you should refer to the original 
work:

"Molina, D., LaTorre, A. Herrera, F. SHADE with Iterative Local Search for
Large-Scale Global Optimization. Proceeding of the 2018, IEEE Congress on Evolutionary
Computation, Rio de Janeiro, Brasil, 8-13 July, 2018, pp 1252-1259"

It was presented in the WCCI 2018, in particular in the IEEE Congress on
Evolutionary Computation. The
[slides are available](https://speakerdeck.com/dmolina/shade-with-iterative-local-search-for-large-scale-global-optimization).

## Install ##

It is recommended to use

```shell
source install.sh
```

That command will create a virtual environment (virtualenv) in the directory 
venv with all required dependencies. 

## Run ##

The source code is prepared for doing the experiments using the Large-Scale
Global Optimization CEC'2013 benchmark.

Parameters:

python shadeils -f <function> -s <seed> [-r <run>] ...

- **function** is the number of function (between 1-15).
- **run** is the number of run for evaluations. 
- **seed** is a seed value (integer value between 1 and 5).

There are other optional parameters, you can run

```shell
python shadeils.py -h```

to get the descriptions of the different optional parameters.
