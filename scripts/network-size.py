'''
network-size.py
calculate the size of a convolutional neural network
'''

# 1. Set variables
print("Input variables:")
in_x = int(input("\tWhat is the x dimension of the input?\n\t "))
in_y = int(input("\tWhat is the y dimension of the input?\n\t "))
in_z = int(input("\tWhat is the z/depth dimension of the input?\n\t "))
batch_size = int(input("\tHow many samples in each minibatch?\n\t "))

print("\nNetwork variables:")
n_layers = int(input("\tHow many convolutional layers?\n\t "))
filter_size = int(input("\tHow many weights per filter? (ex. 3x3x3 = 27)\n\t "))
layer_neurons = [0]*n_layers
layer_strides = [0]*n_layers
layer_output = [0]*n_layers
for i in range(n_layers):
    layer_neurons[i] = int(input("\tHow many neurons in layer {}?\n\t ".format(i)))
for i in range(n_layers):
    layer_strides[i] = int(input("\tWhat stride is used in layer {}?\n\t ".format(i)))

n_fc_layers = int(input("\tHow many fully connected layers?\n\t "))
if (n_fc_layers == 2):
    n_fc_neurons = int(input("\tHow many neurons in first fully connected layer?\n\t "))
if (n_fc_layers > 2):
    print("Can't calculate that yet")
    exit()


# 2. Do the calculation
sample = in_x * in_y * in_z
batch = sample * batch_size
layer_params = [0]*n_layers
layer_params[0] = layer_neurons[0] * filter_size
s0 = layer_strides[0]
layer_output[0] = int(layer_neurons[0] * (in_x / s0) + (in_y / s0) + (in_z / s0))
for i in range(1, n_layers):
    s = layer_strides[i]
    layer_params[i] = layer_params[i-1] * filter_size
    layer_output[i] = int(layer_output[i-1] * (in_x / s) + (in_y / s) + (in_z / s))


# 3. Print the information
print("\nEach sample will have {} 32-bit floats as raw input".format(sample))
print("Each batch has {} 32-bit floats as raw input".format(batch))
for i in range(n_layers):
    print("Layer {}".format(i))
    print("\t{} parameters".format(layer_params[i]))
    print("\t{} variables in output volume".format(layer_output[i]))
print("----------------------------------")
print("Total network parameters: {}".format(sum(layer_params)))
print("Total variables per batch: {}".format(sum(layer_output) * batch_size))

approx_total = (2*sum(layer_params)) + (sum(layer_output) * batch_size)
num_bytes = approx_total * 4  #32 bits (4 bytes) per float
print("Approximate megabytes required: {}".format(num_bytes / 1000000))



