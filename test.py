import tensornetwork as tn

import numpy as np
#import tensornetwork as tn

# Create the nodes
from examples.fft import fft_test

tn.set_default_backend("tdd")
fft_test.test_fft()