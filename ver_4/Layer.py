import numpy as np

#  regular data shape (batch, input_Z, output_Z)
#  conv data shape (batch, W, H, Channer)
#  tensers data shape (batch, num of word, tenser lenght)

class Layer:
    def __init__(self, inP, outP, active_func=None, optimazer=None, trainable=True, normalization=False, dropout=False, leaning_rate=0.01):
        self.param_data_memory = {
            "input":[],
            "linear":[],
            "norm":[],
            "activation":[],
            "dropout":[],
            "wight":[],
            "bias":[],
            "gamma":[],
            "beta":[],
        }

        #learinig rate :)
        self.learning_rate = leaning_rate
        
        #data memory
        self.input_data = None
        self.pre_activ_output = None
        self.norm_data_output = None
        self.activ_output = None
        self.dropted_data_output = None

        #norma outputs
        self.mu = None
        self.ver = None
        self.hat_z = None

        #parameter memory
        self.in_shape = inP
        self.out_shape = outP

            #params
        self.Wight = None
        self.Bias = None
        self.gamma = None
        self.beta = None

        self.dropout_mask = None

        #gradient
        self.D_input = None

            #D params
        self.D_Wight = None
        self.D_Bias = None
        self.D_gamma = None
        self.D_beta = None

        #functions
        self.activation_func = active_func
        self.optimazer = optimazer

        #other
        self.trainable = trainable
        self.normaliz = normalization
        self.drop_out = dropout

    def cahck_param_data(self):
        self.param_data_memory["input"]=self.input_data
        self.param_data_memory["linear"]=self.pre_activ_output
        self.param_data_memory["norm"]=self.norm_data_output
        self.param_data_memory["activation"]=self.activ_output
        self.param_data_memory["dropout"]=self.dropted_data_output
        self.param_data_memory["wight"]=self.Wight
        self.param_data_memory["bias"]=self.Bias
        self.param_data_memory["gamma"]=self.gamma
        self.param_data_memory["beta"]=self.beta

    def Linear_prosecc(self, input):
        return np.matmul(input, self.Wight) + self.Bias
    
    def Normalization(self, input, epsilon=1e-9):
        self.mu = np.mean(input, axis=0)
        self.ver = np.var(input, axis=0)

        self.hat_z = (input - self.mu) / np.sqrt(self.ver + epsilon)

        return (self.gamma * self.hat_z) + self.beta  

    def dropout(self, input, p_):
        if self.trainable:
            mask = np.random.choice([0,1], input.shape, p=[p_, 1-p_]).astype(float)
            output = (mask * input) / (1 - p_)
            return output, mask
        else:
            return input, None
    
    def active_data(self, input, active_func):
        return active_func.activ(input)

    def build_layer(self):
        self.Wight = np.random.randn(self.out_shape, self.in_shape) * np.sqrt(2. / self.in_shape)
        self.Bias = np.random.randn(1, self.out_shape)

        self.gamma = np.ones((1, self.out_shape))
        self.beta = np.zeros((1, self.out_shape))

    def farword(self, input_data):
        self.input_data = input_data

        out = self.Linear_prosecc(input_data)
        self.pre_activ_output = out.copy()

        if self.normaliz:
            out = self.Normalization(out)
            self.norm_data_output = out.copy()
        
        if self.activation_func != None:
            out = self.active_data(out)
            self.activ_output = out.copy()

        if self.drop_out:
            out, mask = self.dropout(out)
            self.dropted_data_output = out.copy()
            self.dropout_mask = mask.copy()

        return out
    
    def D_Linear_prosecc(self, D_input):
        self.D_Wight = np.matmul(D_input, self.input_data)
        self.D_Bias = np.sum(D_input, axis=0, keepdims=True)
        return np.matmul(D_input, self.Wight.T)
    
    def D_active_data(self, D_input):
        return D_input * self.activation_func.derev(self.pre_activ_output)

    def D_Normalization(self, D_input, epsilon=1e-9):
        N = D_input.shape[0]

        # шаги как в теории
        dx_hat = D_input * self.gamma

        dvar = np.sum(dx_hat * (self.pre_activ_output - self.mu) * -0.5 * (self.ver + epsilon) ** (-1.5), axis=0)
        dmu = np.sum(dx_hat * (-1 / np.sqrt(self.ver + epsilon)), axis=0) + dvar * np.mean(-2 * (self.pre_activ_output - self.mu), axis=0)

        dx = dx_hat / np.sqrt(self.ver + epsilon) + dvar * 2 * (self.pre_activ_output - self.mu) / N + dmu / N

        self.D_gamma = np.sum(D_input * self.hat_z, axis=0)
        self.D_beta = np.sum(D_input, axis=0)

        return dx
    
    def D_dropout(self, D_input):
        return D_input * self.dropout_mask
    
    def backword(self, D_input):
        D_inp = D_input

        if self.drop_out:
            D_inp= self.D_dropout(D_inp)

        if self.activation_func != None:
            D_inp = self.D_active_data(D_inp)

        if self.normaliz:
            D_inp = self.D_Normalization(D_inp)
        
        D_inp = self.D_Linear_prosecc(D_inp)

        self.tune_Param()
        self.cahck_param_data()
        return D_inp


    def tune_Param(self):
        if self.D_Wight != None:
            updata_param = self.optimazer.optim(self.Wight ,self.D_Wight)
            self.Wight -= updata_param * self.learning_rate

        if self.D_Bias != None:
            updata_param = self.optimazer.optim(self.Bias, self.D_Bias)
            self.Bias -= updata_param * self.learning_rate

        if self.D_gamma != None:
            updata_param = self.optimazer.optim(self.gamma, self.D_gamma)
            self.gamma -= updata_param * self.learning_rate

        if self.D_beta != None:
            updata_param = self.optimazer.optim(self.beta, self.D_beta)
            self.beta -= updata_param * self.learning_rate


# - - - - - - - - testing - - - - - - - - - - - 

        #data shape (batch,  input fichur)
data = np.array([[1,2,3],
                 [1,2,3],
                 [1,2,3]])

layer_1 = Layer(3,1, normalization=True, dropout=True)

layer_1.build_layer()
out = layer_1.farword(data)

D_data = np.array([[1,1,1],
                   [1,1,1],
                   [1,1,1]])

D_out = layer_1.backword(D_data)