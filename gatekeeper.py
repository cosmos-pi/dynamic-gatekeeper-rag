import numpy as np


class DynamicGatekeeper:
    """
    Batch-Relative Filtering Algorithm
    """

    def __init__(self, alpha=0.5, min_keep=2):
        """
        alpha: strictness factor
        min_keep: minimum documents to retain
        """
        self.alpha = alpha
        self.min_keep = min_keep

    def filter(self, scores):
        """
        Parameters
        ----------
        scores : list or numpy array
            Judge model confidence scores

        Returns
        -------
        mask : list[bool]
            True = Keep, False = Discard
        """

        scores = np.array(scores, dtype=float)

        mean = np.mean(scores)
        std = np.std(scores)

        # Avoid division by zero
        eps = 1e-8

        # Dynamic threshold
        threshold = mean + self.alpha * std

        # Initial filtering
        keep_mask = scores >= threshold

        # ---- Signal Salvaging ----
        if keep_mask.sum() < self.min_keep:
            top_indices = np.argsort(scores)[-self.min_keep:]
            keep_mask = np.zeros_like(scores, dtype=bool)
            keep_mask[top_indices] = True

        return keep_mask.tolist()


# ---------------- Example Usage ----------------
if __name__ == "__main__":

    gatekeeper = DynamicGatekeeper(alpha=0.5, min_keep=2)

    easy_batch = [0.92, 0.95, 0.91, 0.88, 0.93]
    hard_batch = [0.45, 0.42, 0.48, 0.40, 0.44]

    print("Easy Batch:", gatekeeper.filter(easy_batch))
    print("Hard Batch:", gatekeeper.filter(hard_batch))

