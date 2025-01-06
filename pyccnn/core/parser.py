from ast import literal_eval
import configparser
import importlib
import pkgutil
import os

from pyccnn.core.units.unit import CCNNUnit
from pyccnn.core import losses, model, monitor


def _parse_config_string(value):
    config = configparser.RawConfigParser()
    config.optionxform = str 
    config.read_string(value)
    args = {section: {k: literal_eval(v) for (k, v) in config.items(section)}
            for section in config.sections()}
    return config, args


def _parse_config_file(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f'File {path} does not exist.')
    config = configparser.RawConfigParser()
    config.optionxform = str 
    config.read(path)
    args = {section: {k: literal_eval(v) for (k, v) in config.items(section)}
            for section in config.sections()}
    return config, args


def _load_config(config, args):
    output_args = {k: args['Output'][k] for (k, _) in config.items('Output')}
    candidate_args = {k: args['Candidate'][k] for (k, _) in config.items('Candidate')}
    network_args = {k: args['Network'][k] for (k, _) in config.items('Network')}
    train_args = {k: args['Train'][k] for (k, _) in config.items('Train')}

    # parse output neuron arguments
    unit_modules = [importlib.import_module(f"pyccnn.core.units.{module_info.name}")
                    for module_info in pkgutil.iter_modules(
                        importlib.import_module('pyccnn.core.units').__path__)]
    output_type = CCNNUnit.getattr_any(unit_modules, output_args.pop('class'))
    output_args = output_type.parse_args(output_args)

    # parse candidate neuron arguments
    candidate_type = CCNNUnit.getattr_any(unit_modules, candidate_args.pop('class'))
    candidate_args = candidate_type.parse_args(candidate_args)
    
    # parse network arguments
    network_args['output_unit'] = output_type(**output_args)
    network_args['candidate_unit'] = candidate_type(**candidate_args)
    if 'metric_function_args' in network_args:
        metric_fn_factory = getattr(losses, network_args['metric_function'])
        metric_fn_args = network_args.pop('metric_function_args')
        network_args['metric_function'] = metric_fn_factory(**metric_fn_args)
    else:
        network_args['metric_function'] = getattr(losses, network_args['metric_function'])
    if 'output_connection_types' in network_args:
        network_args['output_connection_types'] = getattr(model, network_args['output_connection_types'])
    if 'candidate_connection_types' in network_args:
        network_args['candidate_connection_types'] = getattr(model, network_args['candidate_connection_types'])

    # parse train arguments
    stopping_rule_train_type = getattr(monitor, train_args['stopping_rule'])
    stopping_rule_train_kwargs = train_args.pop('stopping_rule_kwargs', {})
    train_args['stopping_rule'] = stopping_rule_train_type(**stopping_rule_train_kwargs)

    return network_args, train_args


def load_config(path):
    return _load_config(*_parse_config_file(path))


def load_config_from_string(path):
    return _load_config(*_parse_config_string(path))
