
import yaml



def load_config(config_path='config.yml'):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as config_file:
            ConfigManager.CONFIG = yaml.safe_load(config_file)
            return {}
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return {}


class ConfigManager:
    CONFIG = None

