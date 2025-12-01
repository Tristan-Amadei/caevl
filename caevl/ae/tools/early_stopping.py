class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.

    Parameters
    ----------
    patience : int, optional
        Number of epochs to wait after last improvement before stopping the training. Default is 5.
    min_delta : float, optional
        Minimum change in the monitored quantity to qualify as an improvement. Can be negative. Default is 0.
    """

    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        """
        Check if training should be stopped.

        Parameters
        ----------
        val_loss : float
            Current validation loss.

        Returns
        -------
        None
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss <= self.best_loss + self.min_delta:
            self.best_loss = min(val_loss, self.best_loss)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
