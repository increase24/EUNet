from .EUNet import EUNet

def get_network(name, param):
    model = {'EUNet':EUNet}[name]
    return model(param)
