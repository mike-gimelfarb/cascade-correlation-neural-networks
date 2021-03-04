class CCNNUnit:
    
    def build(self, num_inputs, num_outputs, num_targets):
        raise NotImplementedError
    
    def evaluate_losses(self, X, y):
        raise NotImplementedError

    def train(self, X, y):
        raise NotImplementedError
    
    def finalize(self, sess, inputs, index=-1):
        raise NotImplementedError
    
    def clear(self):
        raise NotImplementedError
