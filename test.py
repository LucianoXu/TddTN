import tensornetwork as tn

import numpy as np
#import tensornetwork as tn

# Create the nodes

tn.set_default_backend("tdd")
a = tn.Node(np.ones((10,)))
b = tn.Node(np.ones((10,)))
edge = a[0] ^ b[0] # Equal to tn.connect(a[0], b[0])
final_node = tn.contract(edge)
print(final_node.tensor) # Should print 10.0