import json


'''
    # Conv : (num_filters, filter_size, stride)
    'conv1' : (32, 3, 1)
    # Pool : (filter_size, stride)
    'pool1' = (2, 2)
    # Dense : (num_neurons)
    'dense1' = (1024)
'''

def generateModelFile():
    model = [ ('conv1', (32, 3, 1) ),
              ('pool1', (2, 2) ),
              ('conv2', (64, 3, 1) ),
              ('pool2', (2, 2)), 
              ('dense1', (512)),
              ('dense2', (256)) ]

    with open('models/cnn.json', 'w') as outfile:
         json.dump(model, outfile, sort_keys = True, indent = 4, ensure_ascii = False)

def loadModelFile(modelfile):
    with open(modelfile) as data_file:
        model = json.load(data_file)
    arch = []
    conv_sizes = []
    dense_sizes = []
    for layer, spec in model:
        layer_name = layer[ : -1]




if __name__ == '__main__':
    loadModelFile('models/cnn.json')

