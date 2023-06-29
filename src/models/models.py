from typing import Dict

from src.models import (
    lcnn,
    specrnet,
    whisper_specrnet,
    rawnet3,
    whisper_lcnn,
    meso_net,
    whisper_meso_net
)


def get_model(model_name: str, config: Dict, device: str):
    if model_name == "rawnet3":
        return rawnet3.prepare_model()
    elif model_name == "lcnn":
        return lcnn.FrontendLCNN(device=device, **config)
    elif model_name == "specrnet":
        return specrnet.FrontendSpecRNet(
            device=device,
            **config,
        )
    elif model_name == "mesonet":
        return meso_net.FrontendMesoInception4(
            input_channels=config.get("input_channels", 1),
            fc1_dim=config.get("fc1_dim", 1024),
            frontend_algorithm=config.get("frontend_algorithm", "lfcc"),
            device=device,
        )
    elif model_name == "whisper_lcnn":
        return whisper_lcnn.WhisperLCNN(
            input_channels=config.get("input_channels", 1),
            freeze_encoder=config.get("freeze_encoder", False),
            device=device,
        )
    elif model_name == "whisper_specrnet":
        return whisper_specrnet.WhisperSpecRNet(
            input_channels=config.get("input_channels", 1),
            freeze_encoder=config.get("freeze_encoder", False),
            device=device,
        )
    elif model_name == "whisper_mesonet":
        return whisper_meso_net.WhisperMesoNet(
            input_channels=config.get("input_channels", 1),
            freeze_encoder=config.get("freeze_encoder", True),
            fc1_dim=config.get("fc1_dim", 1024),
            device=device,
        )
    elif model_name == "whisper_frontend_lcnn":
        return whisper_lcnn.WhisperMultiFrontLCNN(
            input_channels=config.get("input_channels", 2),
            freeze_encoder=config.get("freeze_encoder", False),
            frontend_algorithm=config.get("frontend_algorithm", "lfcc"),
            device=device,
        )
    elif model_name == "whisper_frontend_specrnet":
        return whisper_specrnet.WhisperMultiFrontSpecRNet(
            input_channels=config.get("input_channels", 2),
            freeze_encoder=config.get("freeze_encoder", False),
            frontend_algorithm=config.get("frontend_algorithm", "lfcc"),
            device=device,
        )
    elif model_name == "whisper_frontend_mesonet":
        return whisper_meso_net.WhisperMultiFrontMesoNet(
            input_channels=config.get("input_channels", 2),
            fc1_dim=config.get("fc1_dim", 1024),
            freeze_encoder=config.get("freeze_encoder", True),
            frontend_algorithm=config.get("frontend_algorithm", "lfcc"),
            device=device,
        )
    else:
        raise ValueError(f"Model '{model_name}' not supported")
