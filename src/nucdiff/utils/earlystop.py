class EarlyStopper:
    def __init__(self, patience=1):
        self.best = float("inf"); self.pat = patience; self.counter = 0
    def step(self, metric):
        if metric < self.best:
            self.best = metric; self.counter = 0
        else:
            self.counter += 1
        return self.counter > self.pat
