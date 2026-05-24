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
        if prev_loss == curr_loss:
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
            self.history.append((step, val_loss))
            return crossed_now
            
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
        
    def get_crossed_bands(self) -> List[float]:
        """Return list of bands that have been crossed."""
        return [band for band, crossed in self.crossed_bands.items() if crossed]
        
    def get_remaining_bands(self) -> List[float]:
        """Return list of bands that have not yet been crossed."""
        return [band for band, crossed in self.crossed_bands.items() if not crossed]


def compute_primary_band(mlm_summary_df, architecture_list: List[str], size: str) -> Optional[float]:
    """
    Determine the primary comparison band for a given model size.
    The primary band is the lowest (deepest) validation loss band that ALL
    architectures in the architecture_list have successfully crossed for this size.
    """
    if mlm_summary_df is None or mlm_summary_df.empty:
        return None
        
    # Filter for the specific size
    df_size = mlm_summary_df[mlm_summary_df['Model_Size'] == size]
    if df_size.empty:
        return None
        
    arch_bands = {}
    for arch in architecture_list:
        arch_df = df_size[df_size['Architecture'] == arch]
        # Get unique bands crossed by this architecture (ignore 'None' or empty)
        bands = arch_df['Band'].dropna().unique()
        # Convert to float and filter out any token markers like '50M'
        float_bands = []
        for b in bands:
            try:
                float_bands.append(float(b))
            except ValueError:
                pass
        arch_bands[arch] = set(float_bands)
        
    # Find the intersection of bands crossed by ALL architectures
    if not arch_bands:
        return None
        
    common_bands = set.intersection(*arch_bands.values())
    
    if not common_bands:
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
