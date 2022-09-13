from importlib.metadata import distribution
from operator import index
import pickle
import time
from typing import Callable, Dict, cast
import torch as t
from runner import Ladder, LadderConfig
import gaussian_regression as gr
from model import LadderModel
import gaussian_regression
import logging
import scipy.optimize

logging.getLogger("matplotlib.font_manager").disabled = True

logging.basicConfig(filename='out.txt', encoding='utf-8', level=logging.DEBUG)

logging.basicConfig()


normal = t.distributions.Normal(0, 1)


class BayesOpt:
    def __init__(self, d: Dict[str, tuple[float, float]], model: Callable[[Dict[str, float]], t.Tensor], name=None):
        self.name = "BayesOpt_{}".format(name if name is not None else time.time())

        self.model = model
        self.init_device()
        self.init_hp_ranges(d)
        self.init_X()
        self.init_y()

    def init_device(self):
        self.device = t.device("cpu")

    def init_hp_ranges(self, d: Dict[str, tuple[float, float]]):
        self.hp_ranges = d

        self.key_to_index = dict()
        self.index_to_key = dict()

        self.num_hps = len(self.hp_ranges)

        self.hp_lb = t.zeros(self.num_hps)
        self.hp_ub = t.zeros(self.num_hps)

        for i, key in enumerate(self.hp_ranges.keys()):
            self.key_to_index[key] = i
            self.index_to_key[i] = key

            self.hp_lb[i] = self.hp_ranges[key][0]
            self.hp_ub[i] = self.hp_ranges[key][1]

    def init_X(self):
        num_start_points = 4

        x_min = t.zeros(self.num_hps, dtype=t.float64)
        x_max = t.zeros(self.num_hps, dtype=t.float64)

        for i in range(self.num_hps):
            x_min[i] = self.hp_ranges[self.index_to_key[i]][0]
            x_max[i] = self.hp_ranges[self.index_to_key[i]][1]

        res = t.rand((num_start_points, self.num_hps))
        res = res * (x_max - x_min) + x_min

        self.X = res

    def calculate_y(self, x: t.Tensor):
        d = self.get_hp_dict(x)

        logging.info("evaluating with hyperparameters: {}".format(d))
        res = self.model(d)
        logging.info("hyperparameters {} produced score {}".format(d, res.item()))

        return res

    def init_y(self):
        res = t.zeros(self.X.shape[0], dtype=t.float64, device=self.device)
        for i in range(self.X.shape[0]):
            res[i] = self.calculate_y(self.X[i])

        self.y = res

    def get_hp_dict(self, x: t.Tensor):
        hp_dict = dict()
        for i, hp in enumerate(x):
            hp_dict[self.index_to_key[i]] = float(hp)

        return hp_dict

    def get_gr_model(self):
        # find priors
        priors = gaussian_regression.find_priors(self.X, self.y)
        logging.info("found priors\tmean: {}\tnoise: {}\tv: {}\tl: {}".format(priors[0].item(), priors[1].item(), priors[2].item(), priors[3].item()))
        # return the model
        res = gaussian_regression.model(self.X, self.y, *priors)
        # pickle.dump(res, open("./models/{}.p".format(int(time.time())), "wb"))
        res.dump(open("./models/{}_regression_model.p".format(self.name), "wb"))
        return res

    def expected_improvement(self, ys: t.Tensor, x: t.Tensor, gr_model: gaussian_regression.model, eps=None):
        if eps is None:
            eps = 0.05 * ys

        mean, std = gr_model(x)
        if std.dim() == 2:
            std = std.diagonal()
        std = std ** 0.5

        z_demeaned = (mean - ys - eps)

        pdf = normal.log_prob(z_demeaned / std).exp()
        cdf = normal.cdf(z_demeaned / std)

        ei = std * pdf + z_demeaned * cdf

        return ei

    def max_expected_improvement(self, epochs=100, lr=0.001, eps=None):
        ys = self.y.max()
        xs = self.X[self.y.argmax()]
        x = (self.X.mean(dim=0)).clone().requires_grad_()

        gr_model = self.get_gr_model()

        # want to minimize this function
        def ei_fn(x: t.Tensor):
            return - self.expected_improvement(ys, x, gr_model, eps)

        optim = t.optim.LBFGS([x], line_search_fn="strong_wolfe")

        def closure():
            optim.zero_grad()
            loss = ei_fn(x)
            loss.backward()
            return loss

        for i in range(100):
            optim.step(closure)

        x = self.clamp_hp(x)

        logging.debug("best so far: {} with score {}".format(self.get_hp_dict(xs), ys))
        logging.debug("found hp: {} with expected improvement of {}".format(self.get_hp_dict(x), -ei_fn(x).item()))

        return x.clone().detach()

    def add_new_x(self, x: t.Tensor):
        self.X = t.concat([self.X, x.unsqueeze(0)], dim=0)
        y = self.calculate_y(x)
        self.y = t.concat([self.y, y.unsqueeze(0)])

        self.dump(open("./models/{}.p".format(self.name), "wb"))

    def find_hyperparameters(self, epochs=100):
        for i in range(epochs):
            new_x = self.max_expected_improvement()
            self.add_new_x(new_x)

    def clamp_hp(self, x: t.Tensor):
        x_out = x.clone()
        for i in range(x.shape[0]):
            x_out[i] = max(min(x[i].item(), self.hp_ranges[self.index_to_key[i]][1]), self.hp_ranges[self.index_to_key[i]][0])

        return x_out

    def dump(self, file):
        pickle.dump(self, file)

    @classmethod
    def load(cls, file):
        return cast(cls, pickle.load(file))
