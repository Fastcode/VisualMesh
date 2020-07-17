import training.dataset as ds

from .merge_configuration import merge_configuration


def Dataset(config, split, batch_size=None):

    dataset_config = config["dataset"][split]
    specific_config = merge_configuration(config, dataset_config.get("config", {}))

    # Datasets work out the flavour themselves provided they are given the correct configuration
    return ds.Dataset(
        paths=dataset_config["paths"],
        batch_size=batch_size if batch_size is not None else dataset_config["batch_size"],
        keys=dataset_config.get("keys", {}),
        **specific_config,
    )
