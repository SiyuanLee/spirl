from contextlib import contextmanager
import itertools
import copy
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import torch.nn.functional as F

from spirl.components.base_model import BaseModel
from spirl.components.logger import Logger
from spirl.modules.losses import KLDivLoss, NLL, CELoss
from spirl.modules.subnetworks import BaseProcessingLSTM, Predictor, Encoder
from spirl.modules.recurrent_modules import RecurrentPredictor
from spirl.utils.general_utils import batch_apply, AttrDict, ParamDict, split_along_axis, get_clipped_optimizer
from spirl.utils.pytorch_utils import map2np, ten2ar, RemoveSpatial, ResizeSpatial, map2torch, find_tensor, \
                                        TensorModule, RAdam
from spirl.utils.vis_utils import fig2img
from spirl.modules.variational_inference import ProbabilisticModel, Gaussian, MultivariateGaussian, get_fixed_prior, \
                                                mc_kl_divergence
from spirl.modules.layers import LayerBuilderParams
from spirl.modules.mdn import MDN, GMM
from spirl.modules.flow_models import ConditionedFlowModel
from spirl.models.vq_layer_kmeans import Quantize


class ClsVQMdl(BaseModel, ProbabilisticModel):
    """Skill embedding + prior model for SPIRL algorithm."""
    def __init__(self, params, logger=None):
        BaseModel.__init__(self, logger)
        ProbabilisticModel.__init__(self)
        self._hp = self._default_hparams()
        self._hp.overwrite(params)  # override defaults with config file
        self._hp.builder = LayerBuilderParams(self._hp.use_convs, self._hp.normalization)
        self.device = self._hp.device

        self.build_network()
        self.warmup_epoch = 10   # warm up without the VQ layer for 10 epochs

    # not use now
    @contextmanager
    def val_mode(self):
        self.switch_to_prior()
        yield
        self.switch_to_inference()

    def _default_hparams(self):
        # put new parameters in here:
        default_dict = ParamDict({
            'use_convs': False,
            'device': None,
            'n_rollout_steps': 10,        # number of decoding steps
            'cond_decode': False,         # if True, conditions decoder on prior inputs
        })

        # Network size
        default_dict.update({
            'state_dim': 1,             # dimensionality of the state space
            'action_dim': 1,            # dimensionality of the action space
            'nz_enc': 32,               # number of dimensions in encoder-latent space
            'nz_vae': 10,               # number of dimensions in vae-latent space
            'nz_mid': 32,               # number of dimensions for internal feature spaces
            'nz_mid_lstm': 128,         # size of middle LSTM layers
            'n_lstm_layers': 1,         # number of LSTM layers
            'n_processing_layers': 3,   # number of layers in MLPs
        })

        # Learned prior
        default_dict.update({
            'n_prior_nets': 1,              # number of prior networks in ensemble
            'num_prior_net_layers': 6,      # number of layers of the learned prior MLP
            'nz_mid_prior': 128,            # dimensionality of internal feature spaces for prior net
            'nll_prior_train': True,        # if True, trains learned prior by maximizing NLL
            'learned_prior_type': 'gauss',  # distribution type for learned prior, ['gauss', 'gmm', 'flow']
            'n_gmm_prior_components': 5,    # number of Gaussian components for GMM learned prior
        })

        # Loss weights
        default_dict.update({
            'reconstruction_mse_weight': 1.,    # weight of MSE reconstruction loss
            'kl_div_weight': 1.,                # weight of KL divergence loss
            'target_kl': None,                  # if not None, adds automatic beta-tuning to reach target KL divergence
        })

        # add new params to parent params
        parent_params = super()._default_hparams()
        parent_params.overwrite(default_dict)
        return parent_params

    def build_network(self):
        """Defines the network architecture (encoder aka inference net, decoder, prior)."""
        assert not self._hp.use_convs   # currently only supports non-image inputs
        assert self._hp.cond_decode  # need to decode based on state for closed-loop low-level
        self.q = self._build_inference_net()
        print("state_dim", self.state_dim)
        print("z_dim", self._hp.nz_vae)
        self.decoder = Predictor(self._hp,
                                 input_size=self.enc_size + self._hp.nz_vae,
                                 output_size=self._hp.action_dim,
                                 mid_size=self._hp.nz_mid_prior)
        self.codebook_size = 16
        self.vq_layer = Quantize(self._hp.nz_vae, self.codebook_size)
        self.p = self._build_prior_ensemble()

    def _build_prior_ensemble(self):
        assert self._hp.n_prior_nets == 1
        return nn.ModuleList([self._build_prior_net() for _ in range(self._hp.n_prior_nets)])

    def _build_prior_net(self):
        """Supports building Gaussian, GMM and Flow prior networks. Default is Gaussian skill prior."""
        return Predictor(self._hp, input_size=self.prior_input_size, output_size=self.codebook_size,
                         num_layers=self._hp.num_prior_net_layers, mid_size=self._hp.nz_mid_prior)

    def forward(self, inputs, use_learned_prior=False):
        """Forward pass of the SPIRL model.
        :arg inputs: dict with 'states', 'actions', 'images' keys from data loader
        :arg use_learned_prior: if True, decodes samples from learned prior instead of posterior, used for RL
        """
        output = AttrDict()
        inputs.observations = inputs.actions    # for seamless evaluation

        # run inference
        output.z, output.vq_loss, output.vq_index = self._run_inference(inputs)
        # print("vq_index", output.vq_index)

        # infer learned skill prior
        output.q_hat = self.compute_learned_prior(self._learned_prior_input(inputs))
        softmax_q_hat = F.softmax(output.q_hat, dim=1)

        # sample latnet variable
        if self._sample_prior:
            z_index = softmax_q_hat.multinomial(num_samples=1, replacement=True)
            z_index = z_index.reshape(-1)
            # print("z_index", z_index)
            output.z = self.vq_layer.embed_code(z_index)
            # print("prior z", output.z.shape)

        # decode
        assert self._regression_targets(inputs).shape[1] == self._hp.n_rollout_steps
        output.reconstruction = self.decode(output.z,
                                            cond_inputs=self._learned_prior_input(inputs),
                                            steps=self._hp.n_rollout_steps,
                                            inputs=inputs)
        return output

    def loss(self, model_output, inputs):
        """Loss computation of the SPIRL model.
        :arg model_output: output of SPIRL model forward pass
        :arg inputs: dict with 'states', 'actions', 'images' keys from data loader
        """
        losses = AttrDict()

        # reconstruction loss, assume unit variance model output Gaussian
        losses.rec_mse = NLL(self._hp.reconstruction_mse_weight) \
            (Gaussian(model_output.reconstruction, torch.zeros_like(model_output.reconstruction)),
             self._regression_targets(inputs))

        # vq loss
        losses.vq_loss1 = model_output.vq_loss[0]
        losses.vq_loss2 = model_output.vq_loss[1]

        # learned skill prior loss
        losses.q_hat_loss = self._compute_learned_prior_loss(model_output)

        losses.total = self._compute_total_loss(losses)
        return losses

    def _compute_learned_prior_loss(self, model_output):
        loss = CELoss(breakdown=0)(model_output.q_hat, model_output.vq_index.detach())
        # aggregate loss breakdown for each of the priors in the ensemble
        # loss.breakdown = torch.stack([chunk.mean() for chunk in torch.chunk(loss.breakdown, self._hp.n_prior_nets)])
        return loss

    def _log_outputs(self, model_output, inputs, losses, step, log_images, phase, logger, **logging_kwargs):
        """Optionally visualizes outputs of SPIRL model.
        :arg model_output: output of SPIRL model forward pass
        :arg inputs: dict with 'states', 'actions', 'images' keys from data loader
        :arg losses: output of SPIRL model loss() function
        :arg step: current training iteration
        :arg log_images: if True, log image visualizations (otherwise only scalar losses etc get logged automatically)
        :arg phase: 'train' or 'val'
        :arg logger: logger class, visualization functions should be implemented in this class
        """
        self._logger.log_scalar(self.beta, "beta", step, phase)

        # log videos/gifs in tensorboard
        if log_images:
            print('{} {}: logging videos'.format(phase, step))
            self._logger.visualize(model_output, inputs, losses, step, phase, logger, **logging_kwargs)

    def decode(self, z, cond_inputs, steps, inputs=None):
        assert inputs is not None       # need additional state sequence input for full decode
        seq_enc = self._get_seq_enc(inputs)
        decode_inputs = torch.cat((seq_enc[:, :steps], z[:, None].repeat(1, steps, 1)), dim=-1)
        # print("decode_input", decode_inputs.shape)
        return batch_apply(decode_inputs, self.decoder)

    def reset(self):
        """Resets action plan (should be called at beginning of episode when used in RL loop)."""
        self._action_plan = deque()        # stores action plan of LL policy when model is used as policy

    def _build_inference_net(self):
        # condition inference on states since decoder is conditioned on states too
        input_size = self._hp.action_dim + self.prior_input_size
        return torch.nn.Sequential(
            BaseProcessingLSTM(self._hp, in_dim=input_size, out_dim=self._hp.nz_enc),
            torch.nn.Linear(self._hp.nz_enc, self._hp.nz_vae)
        )

    def _run_inference(self, inputs):
        # run inference with state sequence conditioning
        inf_input = torch.cat((inputs.actions, self._get_seq_enc(inputs)), dim=-1)
        z = self.q(inf_input)[:, -1]
        z, loss_vq, vq_index = self.vq_layer(z)
        return z, loss_vq, vq_index

    def compute_learned_prior(self, inputs, first_only=False):
        """Splits batch into separate batches for prior ensemble, optionally runs first or avg prior on whole batch.
           (first_only, avg == True is only used for RL)."""
        # if first_only:
        return self.p[0](inputs)

        # assert inputs.shape[0] % self._hp.n_prior_nets == 0
        # per_prior_inputs = torch.chunk(inputs, self._hp.n_prior_nets)
        # prior_results = [prior(input_batch)
        #                  for prior, input_batch in zip(self.p, per_prior_inputs)]
        #
        # return type(prior_results[0]).cat(*prior_results, dim=0)

    def _learned_prior_input(self, inputs):
        return inputs.states[:, 0]

    def _regression_targets(self, inputs):
        return inputs.actions

    def _get_seq_enc(self, inputs):
        return inputs.states[:, :-1]

    def enc_obs(self, obs):
        """Optionally encode observation for decoder."""
        return obs

    @property
    def resolution(self):
        return 64       # return dummy resolution, images are not used by this model

    @property
    def latent_dim(self):
        return self._hp.nz_vae

    @property
    def state_dim(self):
        return self._hp.state_dim

    @property
    def prior_input_size(self):
        return self.state_dim

    @property
    def n_rollout_steps(self):
        return self._hp.n_rollout_steps

    @property
    def beta(self):
        return self._log_beta().exp()[0].detach() if self._hp.target_kl is not None else self._hp.kl_div_weight


class ImgClsVQMdl(ClsVQMdl):
    """SPiRL model with closed-loop decoder that operates on image observations."""

    def _default_hparams(self):
        default_dict = ParamDict({
            'prior_input_res': 32,  # input resolution of prior images
            'encoder_ngf': 8,  # number of feature maps in shallowest level of encoder
            'n_input_frames': 1,  # number of prior input frames
        })
        # add new params to parent params
        return super()._default_hparams().overwrite(default_dict)

    def _updated_encoder_params(self):
        params = copy.deepcopy(self._hp)
        return params.overwrite(AttrDict(
            use_convs=True,
            use_skips=False,                  # no skip connections needed flat we are not reconstructing
            img_sz=self._hp.prior_input_res,  # image resolution
            input_nc=3*self._hp.n_input_frames,  # number of input feature maps
            ngf=self._hp.encoder_ngf,         # number of feature maps in shallowest level
            nz_enc=self.prior_input_size,     # size of image encoder output feature
            builder=LayerBuilderParams(use_convs=True, normalization=self._hp.normalization)
        ))

    def _build_prior_net(self):
        return nn.Sequential(
            ResizeSpatial(self._hp.prior_input_res),
            Encoder(self._updated_encoder_params()),
            RemoveSpatial(),
            super()._build_prior_net(),
        )

    def _build_inference_net(self):
        self.img_encoder = nn.Sequential(ResizeSpatial(self._hp.prior_input_res),  # encodes image inputs
                                         Encoder(self._updated_encoder_params()),
                                         RemoveSpatial(),)
        return ClsVQMdl._build_inference_net(self)

    def _get_seq_enc(self, inputs):
        # stack input image sequence
        stacked_imgs = torch.cat([inputs.images[:, t:t+inputs.actions.shape[1]]
                                  for t in range(self._hp.n_input_frames)], dim=2)
        # encode stacked seq
        return batch_apply(stacked_imgs, self.img_encoder)

    def _learned_prior_input(self, inputs):
        return inputs.images[:, :self._hp.n_input_frames]\
            .reshape(inputs.images.shape[0], -1, self.resolution, self.resolution)

    def _regression_targets(self, inputs):
        return inputs.actions[:, (self._hp.n_input_frames-1):]

    def unflatten_obs(self, raw_obs):
        """Utility to unflatten [obs, prior_obs] concatenated observation (for RL usage)."""
        assert len(raw_obs.shape) == 2 and raw_obs.shape[1] == self.state_dim \
               + self._hp.prior_input_res**2 * 3 * self._hp.n_input_frames
        return AttrDict(
            obs=raw_obs[:, :self.state_dim],
            prior_obs=raw_obs[:, self.state_dim:].reshape(raw_obs.shape[0], 3*self._hp.n_input_frames,
                                                          self._hp.prior_input_res, self._hp.prior_input_res)
        )

    def enc_obs(self, obs):
        """Optionally encode observation for decoder."""
        return self.img_encoder(obs)

    @property
    def enc_size(self):
        return self._hp.nz_enc

    @property
    def prior_input_size(self):
        return self.enc_size

    @property
    def resolution(self):
        return self._hp.prior_input_res