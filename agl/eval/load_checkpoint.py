"""Checkpoint Loading Utility

Provides a unified interface for loading trained checkpoints with proper
error handling and configuration detection.

This utility handles:
  - Loading checkpoint files (.pt format)
  - Restoring model weights and configuration
  - Automatic device placement
  - Configuration override support
  - Validation of checkpoint integrity

Usage as a module:
  from agl.eval.load_checkpoint import load_checkpoint
  model, cfg, metadata = load_checkpoint('runs/recipe_v3/ckpt_latest.pt')

Usage as a script:
  python -m agl.eval.load_checkpoint --ckpt runs/recipe_v3/ckpt_latest.pt --info
"""
import argparse
import os
from typing import Tuple, Dict, Optional

import torch

from ..config import load_config, Config
from ..models.policy import Policy


def load_checkpoint(
    ckpt_path: str,
    device: str = 'cuda',
    eval_mode: bool = True,
    config_overrides: Optional[Dict] = None
) -> Tuple[Policy, Config, Dict]:
    """Load a trained checkpoint with model and configuration.

    Args:
        ckpt_path: Path to checkpoint file (.pt)
        device: Device to load model onto ('cuda' or 'cpu')
        eval_mode: If True, set model to eval mode and disable gradients
        config_overrides: Optional dict of config values to override

    Returns:
        model: Loaded Policy model
        cfg: Configuration object
        metadata: Dict containing checkpoint metadata (steps, epoch, etc.)

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        KeyError: If checkpoint is missing required keys
        RuntimeError: If model loading fails
    """
    # Validate checkpoint exists
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Load checkpoint
    print(f"Loading checkpoint: {ckpt_path}")
    try:
        ckpt = torch.load(ckpt_path, map_location=device)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")

    # Validate checkpoint structure
    required_keys = ['model', 'cfg']
    missing_keys = [k for k in required_keys if k not in ckpt]
    if missing_keys:
        raise KeyError(f"Checkpoint missing required keys: {missing_keys}")

    # Load configuration
    cfg_dict = ckpt['cfg']
    if config_overrides:
        cfg_dict.update(config_overrides)
    cfg = load_config(overrides=cfg_dict)

    # Create model
    print("Initializing model...")
    model = Policy(cfg).to(device)

    # Load weights
    try:
        model.load_state_dict(ckpt['model'])
    except Exception as e:
        raise RuntimeError(f"Failed to load model weights: {e}")

    # Set eval mode if requested
    if eval_mode:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    # Extract metadata
    metadata = {
        'checkpoint_path': ckpt_path,
        'training_steps': ckpt.get('steps', -1),
        'epoch': ckpt.get('epoch', -1),
        'device': device
    }

    # Add any additional metadata from checkpoint
    for key in ['train_stats', 'git_hash', 'timestamp']:
        if key in ckpt:
            metadata[key] = ckpt[key]

    print(f"✓ Checkpoint loaded successfully")
    print(f"  Training steps: {metadata['training_steps']}")
    print(f"  Device: {metadata['device']}")

    return model, cfg, metadata


def print_checkpoint_info(ckpt_path: str):
    """Print detailed information about a checkpoint file.

    Args:
        ckpt_path: Path to checkpoint file
    """
    if not os.path.exists(ckpt_path):
        print(f"✗ Checkpoint not found: {ckpt_path}")
        return

    print("\n" + "="*70)
    print(f"CHECKPOINT INFO: {ckpt_path}")
    print("="*70)

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location='cpu')

    # File info
    file_size = os.path.getsize(ckpt_path) / (1024 * 1024)  # MB
    print(f"\nFile:")
    print(f"  Path: {ckpt_path}")
    print(f"  Size: {file_size:.2f} MB")

    # Training progress
    print(f"\nTraining Progress:")
    print(f"  Steps: {ckpt.get('steps', 'unknown')}")
    print(f"  Epoch: {ckpt.get('epoch', 'unknown')}")

    # Configuration
    if 'cfg' in ckpt:
        cfg_dict = ckpt['cfg']
        print(f"\nModel Configuration:")

        if 'model' in cfg_dict:
            model_cfg = cfg_dict['model']
            print(f"  use_memory: {model_cfg.get('use_memory', 'N/A')}")
            print(f"  gru_hidden: {model_cfg.get('gru_hidden', 'N/A')}")
            print(f"  reset_between_attempts: {model_cfg.get('reset_between_attempts', 'N/A')}")
            print(f"  use_prev_action: {model_cfg.get('use_prev_action', 'N/A')}")
            print(f"  use_risk_feedback: {model_cfg.get('use_risk_feedback', 'N/A')}")

        if 'task' in cfg_dict:
            task_cfg = cfg_dict['task']
            print(f"\n  Gap Configuration:")
            print(f"    width_lo: {task_cfg.get('width_lo', 'N/A')}")
            print(f"    width_hi: {task_cfg.get('width_hi', 'N/A')}")
            print(f"    roll_max_deg: {task_cfg.get('roll_max_deg', 'N/A')}")

    # Model state
    if 'model' in ckpt:
        model_state = ckpt['model']
        n_params = sum(p.numel() for p in model_state.values())
        print(f"\nModel State:")
        print(f"  Total parameters: {n_params:,}")
        print(f"  State dict keys: {len(model_state)}")

    # Optimizer state (if present)
    if 'optimizer' in ckpt:
        print(f"\nOptimizer State:")
        print(f"  Present: Yes")

    # Additional metadata
    print(f"\nAdditional Data:")
    extra_keys = [k for k in ckpt.keys() if k not in ['model', 'cfg', 'optimizer', 'steps', 'epoch']]
    if extra_keys:
        for key in extra_keys:
            print(f"  {key}: {type(ckpt[key]).__name__}")
    else:
        print(f"  None")

    print("="*70 + "\n")


def find_checkpoints(root_dir: str = './runs') -> list:
    """Find all checkpoint files in a directory tree.

    Args:
        root_dir: Root directory to search

    Returns:
        List of checkpoint paths found
    """
    checkpoints = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.pt'):
                full_path = os.path.join(dirpath, filename)
                checkpoints.append(full_path)
    return sorted(checkpoints)


def main():
    parser = argparse.ArgumentParser(
        description='Checkpoint loading utility',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show checkpoint info
  python -m agl.eval.load_checkpoint --ckpt runs/recipe_v3/ckpt_latest.pt --info

  # List all checkpoints
  python -m agl.eval.load_checkpoint --list

  # Test loading a checkpoint
  python -m agl.eval.load_checkpoint --ckpt runs/recipe_v3/ckpt_latest.pt --test
        """
    )
    parser.add_argument('--ckpt', type=str,
                        help='Path to checkpoint file')
    parser.add_argument('--info', action='store_true',
                        help='Print detailed checkpoint information')
    parser.add_argument('--test', action='store_true',
                        help='Test loading the checkpoint')
    parser.add_argument('--list', action='store_true',
                        help='List all checkpoints in runs/')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for test loading (default: cpu)')
    args = parser.parse_args()

    # List checkpoints
    if args.list:
        print("Searching for checkpoints in ./runs/...")
        checkpoints = find_checkpoints('./runs')
        if checkpoints:
            print(f"\nFound {len(checkpoints)} checkpoint(s):\n")
            for i, ckpt_path in enumerate(checkpoints, 1):
                size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
                print(f"  {i}. {ckpt_path} ({size_mb:.2f} MB)")
        else:
            print("No checkpoints found.")
        return

    # Require --ckpt for other operations
    if not args.ckpt:
        parser.error("--ckpt is required (unless using --list)")

    # Print info
    if args.info:
        print_checkpoint_info(args.ckpt)

    # Test loading
    if args.test:
        print(f"\nTesting checkpoint loading...")
        try:
            model, cfg, metadata = load_checkpoint(
                args.ckpt,
                device=args.device,
                eval_mode=True
            )
            print(f"✓ Test successful!")
            print(f"\nModel info:")
            print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
            print(f"  Hidden size: {cfg.model.gru_hidden}")
            print(f"  Memory enabled: {cfg.model.use_memory}")
            print(f"\nMetadata:")
            for key, value in metadata.items():
                print(f"  {key}: {value}")
        except Exception as e:
            print(f"✗ Test failed: {e}")
            raise

    # If neither --info nor --test specified, just show info
    if not args.info and not args.test:
        print_checkpoint_info(args.ckpt)


if __name__ == '__main__':
    main()
