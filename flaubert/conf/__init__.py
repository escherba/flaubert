from pymaptools.unicode_yaml import yaml
from pkg_resources import resource_filename


CONFIG_FILENAME = resource_filename(__name__, 'default.yaml')
with open(CONFIG_FILENAME, 'r') as fh:
    CONFIG = yaml.load(fh)
