import yaml


def load_datasets_config(config_path: str) -> tuple[dict[str, str], dict[str, str]]:
    """Load and validate datasets configuration.

    Args:
        config_path: Path to the datasets configuration file.
        
    Returns:
        Tuple containing train and test dataset paths dictionaries.

    Raises:
        ValueError: If the configuration file is invalid or missing required sections.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Invalid datasets configuration file format")

    train_datasets = config.get("train", {})
    test_datasets = config.get("test", {})

    if not train_datasets or not test_datasets:
        raise ValueError("Both 'train' and 'test' sections must be present in the datasets configuration")

    return train_datasets, test_datasets

