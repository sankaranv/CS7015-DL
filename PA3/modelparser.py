import json


'''
    # Conv : (num_filters, filter_size, stride)
    'conv1' : (32, 3, 1)
    # Pool : (filter_size, stride)
    'pool1' = (2, 2)
    # Dense : (num_neurons)
    'dense1' = (1024)
'''

def generateModelFile(modelfile):
    model = [ ('input', {'num_neurons' : 784}),
              ('conv1', {'filter_size' : 5, 'num_filters' : 32, 'stride' : 1, 'padding' : 'SAME'}),
              ('conv2', {'filter_size' : 3, 'num_filters' : 64, 'stride' : 1, 'padding' : 'SAME'}),
              ('pool1', {'filter_size' : 2, 'stride' : 2, 'padding' : 'SAME'}),
              ('dropout1', {'prob' : 0.5}),
              ('conv3', {'filter_size' : 3, 'num_filters' : 128, 'stride' : 1, 'padding' : 'SAME'}),
              ('pool2', {'filter_size' : 2, 'stride' : 2, 'padding' : 'SAME'}),
              ('conv4', {'filter_size' : 3, 'num_filters' : 64, 'stride' : 1, 'padding' : 'SAME'}),
              ('reshape', ()),
              ('fc1'  , {'num_neurons' : 512}),
              ('output', {'num_neurons' :10})
            ]

    with open(modelfile, 'w') as outfile:
         json.dump(model, outfile, sort_keys = True, indent = 4, ensure_ascii = False)

def loadArch(modelfile):
    with open(modelfile) as data_file:
        model = json.load(data_file)
    arch = []
    c_conv, c_fc, c_pool = 1, 1, 1
    for (layer, spec) in model:
        arch_ = {}
        arch_['name'] = layer
        arch_['params'] = 'NONE'
        if 'input' in layer:
            out_depth = 1
            spatial = 28
        elif 'conv' in layer:
            weight, bias = 'Wc{}'.format(c_conv), 'bc{}'.format(c_conv)
            in_depth = out_depth
            out_depth = spec['num_filters']
            filter_size = spec['filter_size']
            padding = spec['padding']
            stride = spec['stride']
            c_conv += 1

            ### 
            arch_['params'] = {'weight' : {'shape' : [filter_size, filter_size, in_depth, out_depth], 
                                                   'name' : weight},
                                      'bias' : {'shape' : [out_depth],
                                                 'name' : bias}
                              }
            arch_['padding'] = padding
            arch_['stride']  = stride
                          
        elif 'pool' in layer:
            in_depth = out_depth
            filter_size = spec['filter_size']
            padding = spec['padding']
            stride = spec['stride']           
            spatial /= stride

            ### 
            arch_['padding'] = padding
            arch_['stride']  = stride
                                       
        elif 'reshape' in layer:
            out_depth = spatial * spatial * out_depth

        elif 'dropout' in layer:
            arch_['prob'] = 0.5

        elif 'fc' in layer:
            weight, bias = 'Wd{}'.format(c_fc), 'bd{}'.format(c_fc)
            in_depth = out_depth
            out_depth = spec['num_neurons']
            ###
            arch_['params'] = {'weight' : {'shape' : [in_depth, out_depth], 
                                                   'name' : weight},
                                      'bias' : {'shape' : [out_depth],
                                                'name' : bias}
                              }
            c_fc += 1
        elif 'output' in layer:
            weight, bias = 'Wout', 'bout'
            in_depth = out_depth
            out_depth = spec['num_neurons']
            ###
            arch_['params'] = {'weight' : {'shape' : [in_depth, out_depth], 
                                                   'name' : weight},
                                      'bias' : {'shape' : [out_depth],
                                                'name' : bias}
                              }
            
        arch.append(arch_)

    return arch

if __name__ == '__main__':
    generateModelFile('models/cnn.json')
