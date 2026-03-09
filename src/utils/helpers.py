"""
Utils Module
Fonctions utilitaires pour le projet
"""
import yaml
import os
from pathlib import Path
from loguru import logger
from typing import Dict, Any


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Charge la configuration depuis un fichier YAML
    
    Args:
        config_path: Chemin vers le fichier de configuration
        
    Returns:
        Dictionnaire de configuration
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Fichier de configuration non trouvé: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Configuration chargée depuis {config_path}")
    return config


_logging_configured = False


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """
    Configure le système de logging (une seule fois)
    
    Args:
        log_level: Niveau de log (DEBUG, INFO, WARNING, ERROR)
        log_file: Chemin vers le fichier de log (optionnel)
    """
    global _logging_configured
    
    if _logging_configured:
        return
    
    # Supprimer le handler par défaut
    logger.remove()
    
    # Ajouter un handler pour la console
    logger.add(
        lambda msg: print(msg, end=""),
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
    )
    
    # Ajouter un handler pour le fichier si spécifié
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
            rotation="100 MB"
        )
    
    _logging_configured = True
    logger.info(f"Logging configuré: niveau={log_level}")


def ensure_directories(directories: list):
    """
    Crée les répertoires s'ils n'existent pas
    
    Args:
        directories: Liste des chemins de répertoires à créer
    """
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Répertoire créé/vérifié: {directory}")


if __name__ == "__main__":
    # Test des utils
    setup_logging()
    logger.info("Test des utils")
