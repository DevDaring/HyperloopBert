import os
import shutil
from typing import List, Dict, Optional, Tuple
import logging

# PRE-REGISTERED ENDPOINT: The primary comparison is at matched validation loss
#           (iso-perplexity), NOT at fixed token budget. This removes model quality
#           as an alternative explanation for bias differences.

class IsoBandTracker:
    """
    Tracks validation loss during training and saves model checkpoints when
    pre-defined loss thresholds (bands) are crossed.
    """
    def __init__(self, target_bands: List[float], save_dir: str, logger: Optional[logging.Logger] = None):
        """
        target_bands: List of validation loss targets (e.g., [4.0, 3.7, 3.4, 3.1])
                      Must be strictly decreasing.
        save_dir: Directory to save the models when a band is crossed.
        """
        # Sort bands descending
        self.target_bands = sorted(target_bands, reverse=True)
        self.save_dir = save_dir
        self.logger = logger
        
        # State: track which bands have been crossed
        self.crossed_bands = {band: False for band in self.target_bands}
        
        # History: [(step, val_loss)]
        self.history = []
        
        os.makedirs(self.save_dir, exist_ok=True)
        
    def _interpolate_crossing_step(self, prev_step: int, prev_loss: float, 
                                   curr_step: int, curr_loss: float, band: float) -> int:
        """
        Linearly interpolate the exact step where the band was crossed.
        """
        import math
        if prev_loss == curr_loss or math.isinf(prev_loss):
            return curr_step
            
        fraction = (prev_loss - band) / (prev_loss - curr_loss)
        exact_step = prev_step + fraction * (curr_step - prev_step)
        return int(exact_step)

    def update(self, step: int, val_loss: float, save_callback) -> List[float]:
        """
        Record a new validation loss and check if any uncrossed bands were crossed.
        If crossed, calls save_callback(band_save_path) to save the model.
        Returns a list of bands that were just crossed.
        """
        crossed_now = []

        if not self.history:
            # Treat the pre-training state as loss = +inf so a first validation
            # that is already below a band still fires the snapshot.
            prev_step, prev_loss = step, float('inf')
        else:
            prev_step, prev_loss = self.history[-1]
        self.history.append((step, val_loss))
        
        for band in self.target_bands:
            if not self.crossed_bands[band]:
                # Did we cross the band between prev_loss and val_loss?
                if prev_loss > band and val_loss <= band:
                    self.crossed_bands[band] = True
                    crossed_now.append(band)
                    
                    # Compute interpolated step for reporting/logging
                    exact_step = self._interpolate_crossing_step(prev_step, prev_loss, step, val_loss, band)
                    
                    # Create directory for this band's snapshot
                    band_str = f"{band:.2f}".replace('.', 'p')
                    snapshot_dir = os.path.join(self.save_dir, f"band_{band_str}")
                    os.makedirs(snapshot_dir, exist_ok=True)
                    
                    if self.logger:
                        self.logger.info(f"Iso-loss band {band} crossed! (Prev: {prev_loss:.4f}, Curr: {val_loss:.4f})")
                        self.logger.info(f"Interpolated crossing step: {exact_step}")
                        self.logger.info(f"Saving snapshot to: {snapshot_dir}")
                        
                    # Trigger the caller's save logic
                    save_callback(snapshot_dir)
                    
        return crossed_now
        
    def state_dict(self) -> Dict:
        """Serialisable tracker state for mid-run checkpointing."""
        return {
            'crossed_bands': {str(band): crossed for band, crossed in self.crossed_bands.items()},
            'history': list(self.history),
        }

    def load_state_dict(self, state: Dict) -> None:
        """Restore tracker state saved by state_dict() (resume support)."""
        for band_str, crossed in state.get('crossed_bands', {}).items():
            band = float(band_str)
            if band in self.crossed_bands:
                self.crossed_bands[band] = bool(crossed)
        self.history = [tuple(item) for item in state.get('history', [])]

    def get_crossed_bands(self) -> List[float]:
        """Return list of bands that have been crossed."""
        return [band for band, crossed in self.crossed_bands.items() if crossed]
        
    def get_remaining_bands(self) -> List[float]:
        """Return list of bands that have not yet been crossed."""
        return [band for band, crossed in self.crossed_bands.items() if not crossed]


def compute_primary_band(mlm_summary_df, architecture_list: List[str], size: str,
                         default_stream_count: int = 4,
                         logger_obj=None) -> Optional[float]:
    """
    Determine the primary comparison band for a given model size.

    The primary band is the lowest (deepest) validation-loss band that EVERY
    architecture crossed for EVERY seed that all architectures share. Pooling
    crossings across seeds (the previous behaviour) could select a band where
    no single seed had full architecture coverage, silently degrading the
    paired contrast to n=1.

    Ablation rows (Merge_At set, or Stream_Count set and != default) are
    excluded so stream-count / early-merge arms cannot influence the primary
    band.
    """
    if mlm_summary_df is None or mlm_summary_df.empty:
        return None

    df_size = mlm_summary_df[mlm_summary_df['Model_Size'] == size]
    if df_size.empty:
        return None

    # Exclude ablation identities from primary-band computation
    if 'Merge_At' in df_size.columns:
        df_size = df_size[df_size['Merge_At'].isna()]
    if 'Stream_Count' in df_size.columns:
        df_size = df_size[df_size['Stream_Count'].isna() |
                          (df_size['Stream_Count'] == default_stream_count)]

    # Seeds present for every architecture
    seed_sets = []
    for arch in architecture_list:
        arch_df = df_size[df_size['Architecture'] == arch]
        seed_sets.append(set(arch_df['Seed'].dropna().unique()))
    if not seed_sets:
        return None
    common_seeds = set.intersection(*seed_sets)
    if not common_seeds:
        if logger_obj:
            logger_obj.warning(f"compute_primary_band({size}): no seed shared by "
                               f"all architectures {architecture_list}.")
        return None

    # Bands crossed by every (architecture, seed) combination
    per_combo_bands = []
    for arch in architecture_list:
        for seed in common_seeds:
            combo = df_size[(df_size['Architecture'] == arch) & (df_size['Seed'] == seed)]
            bands = set()
            for b in combo['Band'].dropna().unique():
                try:
                    bands.add(float(b))
                except (TypeError, ValueError):
                    pass
            per_combo_bands.append(bands)

    common_bands = set.intersection(*per_combo_bands) if per_combo_bands else set()
    if not common_bands:
        if logger_obj:
            logger_obj.warning(f"compute_primary_band({size}): no band crossed by "
                               f"every (architecture, seed); walk the band list "
                               f"shallower or extend training.")
        return None

    # The primary band is the lowest loss (minimum value)
    return min(common_bands)

def setup_iso_bands_from_calibration(calibration_df, target_bands: List[float] = None) -> List[float]:
    """
    Optionally adjust default bands based on actual calibration runs.
    If the calibration_df shows models plateauing before the default target bands,
    this returns adjusted bands.
    """
    if target_bands is None:
        target_bands = [4.0, 3.7, 3.4, 3.1]
        
    if calibration_df is None or calibration_df.empty:
        return target_bands
        
    # In a real run, we would analyze the lowest achieved validation loss
    # and adjust the bands if the lowest default band is unreachable.
    # For this implementation, we return the target_bands as-is, leaving
    # adaptive logic as a manual override if needed.
    return target_bands
