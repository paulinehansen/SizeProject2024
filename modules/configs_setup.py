import os
from datetime import date, datetime
from ruamel.yaml import YAML


def get_default_configs_path():
    """
    Returns the default path to the configuration file for the application.

    :return: The absolute path as a string that points to the 'default_configs.yml' file.
    :rtype: str
    """
    cwd = os.getcwd()
    yaml_path = cwd.split('exe')[0] + 'configs/default_configs.yml'
    #yaml_path = cwd + '/configs/default_configs.yml'
    return yaml_path

def get_default_plot_param_path():
    """
    Constructs and returns the filesystem path to the default plotting parameter
    configuration file.

    :return: Full filesystem path to the default plotting parameter configuration file
    :rtype: str
    """
    cwd = os.getcwd()
    yaml_path = cwd.split('exe')[0] + 'configs/default_plot_param.yml'
    

    return yaml_path

def load_default_configs():
    """
    Loads the default configurations from a YAML file. This method reads the
    default configurations file from the predefined path, parses it using
    the safe YAML loader, and configures specific formatting options such
    as indentation, flow style, and width. The loaded configuration is
    then returned as a Python data structure.

    :raises FileNotFoundError: If the YAML file does not exist at the
        specified path.
    :raises yaml.YAMLError: If there is an error during YAML parsing.
    :raises Exception: For any other unexpected errors related to reading
        or processing the file.
    :return: The default configurations loaded from the YAML file.
    :rtype: Any
    """
    yaml = YAML(typ='safe')
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.default_flow_style = False
    yaml.width = 4096
    yaml_path = get_default_configs_path()
    with open(yaml_path, 'r') as file:
        default_configs = yaml.load(file)
    return default_configs

def load_default_plot_params():
    """
    Loads the default plot parameters from a YAML configuration file. This function
    parses the YAML file located at the default plot parameter path and returns its
    contents as a dictionary or similar construct. The YAML loader is configured to
    use safe loading, apply specific indentation, and operate with a wide formatting.

    :return: The loaded default plot parameters from the YAML file.
    :rtype: dict
    """
    yaml = YAML(typ='safe')
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.default_flow_style = False
    yaml.width = 4096
    yaml_path = get_default_plot_param_path()
    with open(yaml_path, 'r') as file:
        default_plot_param = yaml.load(file)
    return default_plot_param

def load_configs(raw_data_folder):
    """
    Loads configuration settings from the latest YAML file within a specified raw data folder.
    The function searches for YAML files in the `configs` subfolder of the given folder. If no
    configuration files are found or an error occurs during loading, default configurations are
    returned instead.

    :param raw_data_folder: Path to the raw data folder containing the `configs` subdirectory
                            with configuration YAML files.
    :type raw_data_folder: str

    :return: A dictionary containing the loaded configurations. If loading fails, it returns
             default configuration values.
    :rtype: dict
    """
    yaml = YAML(typ='safe')
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.default_flow_style = False
    yaml.width = 4096

    configs_folder = raw_data_folder + r'configs/'
    try:
        yaml_names = [f for f in os.listdir(configs_folder) if f.endswith('.yml')]
        yaml_name = sorted(yaml_names, reverse=True)[0]
        yaml_path = os.path.join(configs_folder, yaml_name)
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                configs = yaml.load(f)
            print(f'Loaded {yaml_name}.')

    except:
        print('Could not load configs, returns default configs.')
        configs = load_default_configs()

    return configs

def update_configs(configs):
    """
    Updates user configuration storage with analysis timestamp and saves it into a YAML file.
    This process utilizes the PyYAML library with predefined formatting preferences
    and dynamically generates a filename based on the current date.

    :param configs: A dictionary containing the configuration data, where the key
        'configs_folder' specifies the folder path for storing the updated file.
    :type configs: dict

    :return: Logs a message indicating the updated YAML configuration file path.
    :rtype: None
    """
    today = date.today()
    now = datetime.now()
    yaml = YAML(typ='safe')
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.default_flow_style = False
    yaml.width = 4096
    config_path = configs['configs_folder']
    configs['analysis_time_stamp'] = now.strftime("%m/%d/%Y, %H:%M:%S")
    print(config_path)
    print(today.strftime('%Y-%m-%d'))
    yaml_name = config_path + today.strftime('%Y-%m-%d') + '_configs.yml'
    yaml_path = os.path.join(config_path, yaml_name)
    
    try:
        with open(yaml_path, "w") as outfile:
            yaml.dump(configs, outfile)
    except OSError as e:
        print(e)
        cwd = os.getcwd()
        configs_folder = cwd + '/configs/'
        if not os.path.exists(configs_folder):
            os.makedirs(configs_folder)
            print('created new configs folder under ', configs_folder)
        
        yaml_path = os.path.join(cwd, yaml_name)
        with open(yaml_path, "w") as outfile:
            yaml.dump(configs, outfile)    
           
    return print(f'Updated {yaml_path} file.')

def get_folder_structure(raw_data_folder):
    """
    Creates necessary folder structure for organizing data processing or analysis outputs.

    The function takes a base directory and creates three subdirectories within it
    if they do not already exist. These subdirectories are named `configs`, `results_IF`,
    and `plots`. This function ensures the required directory structure is properly
    set up for further operations. It returns paths to these subdirectories for use
    in subsequent workflows.

    :param raw_data_folder: The base directory as a string where the folder structure
                            will be created. This directory must be a valid path.
    :return: A tuple containing the paths of the created or existing subdirectories:
             `configs_folder`, `results_folder`, and `plots_folder`.
    """
    # create folder structure
    configs_folder = raw_data_folder + r'configs/'
    results_folder = raw_data_folder + r'results_IF/'
    plots_folder = raw_data_folder + r'plots/'

    for folder in [configs_folder, results_folder, plots_folder]:
        if not os.path.exists(folder):
            os.mkdir(folder)

    print(f'Folder organization created under {raw_data_folder}')

    return configs_folder, results_folder, plots_folder


