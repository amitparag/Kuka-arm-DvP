# This files contains utilities for file loading
# Author : Sébastien Kleff
# Date : 09/20/2021

import yaml
import os

# Load a yaml file (e.g. simu config file)
def load_yaml_file(yaml_file):
    '''
    Load config file (yaml)
    '''
    with open(yaml_file) as f:
        data = yaml.load(f)
    return data 

# Load config file
def load_config_file(config_name):
    '''
    Loads YAML config file in demos/config as a dict
    '''
    config_path = os.path.abspath(os.path.join(os.path.abspath(__file__ + "/../../../"), 'config/ocp_params'))
    config_file = config_path+"/"+config_name+".yml"
    config = load_yaml_file(config_file)
    return config

# Get robot properties paths
def kuka_urdf_path():
    return os.path.join(os.path.abspath(__file__ + "/../../../"), 'config/robot_properties_kuka/iiwa.urdf')

def kuka_mesh_path():
    print(os.path.join(os.path.abspath(__file__ + "/../../../"), 'config/robot_properties_kuka'))
    return os.path.join(os.path.abspath(__file__ + "/../../../"), 'config/robot_properties_kuka')

def results_path():
    return os.path.join(os.path.abspath(__file__ + "/../../../"), "results")