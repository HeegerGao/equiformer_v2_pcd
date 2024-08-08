import logging
import time
import math
import numpy as np
import torch
import torch.nn as nn
from pyexpat.model import XML_CQUANT_OPT

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import conditional_grad
from ocpmodels.models.base import BaseModel
from ocpmodels.models.scn.sampling import CalcSpherePoints
from ocpmodels.models.scn.smearing import (
    GaussianSmearing,
    LinearSigmoidSmearing,
    SigmoidSmearing,
    SiLUSmearing,
)

try:
    from e3nn import o3
except ImportError:
    pass

from .gaussian_rbf import GaussianRadialBasisLayer
from torch.nn import Linear
from .edge_rot_mat import init_edge_rot_mat
from .so3 import (
    CoefficientMappingModule,
    SO3_Embedding,
    SO3_Grid,
    SO3_Rotation,
    SO3_LinearV2
)
from .module_list import ModuleListInfo
from .so2_ops import SO2_Convolution
from .radial_function import RadialFunction
from .layer_norm import (
    EquivariantLayerNormArray, 
    EquivariantLayerNormArraySphericalHarmonics, 
    EquivariantRMSNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonicsV2,
    get_normalization_layer
)
from .transformer_block_pcd import (
    SO2EquivariantGraphAttention,
    FeedForwardNetwork,
    TransBlockV2, 
)
from .input_block import EdgeDegreeEmbeddingPCD

@registry.register_model("equiformer_v2_pcd")
class EquiformerV2_PCD(BaseModel):
    """
    Equiformer v2 for point cloud input

    Args to be set by user:
        input_types (list):         List of input feature types
        input_channels (list):      List of input feature channels
        max_output_channels (int):  Maximum number of output channels. E.g., if you want to output 1 channel for type-0 and 3 channels for type-1, you should set this to 3
        max_neighbors (int):        Maximum number of neighbors per point. This is used for constructing the graph
        max_radius (float):         Maximum distance between nieghboring atoms in Angstroms. This is used for the edge embedding kernel
        num_layers (int):           Number of layers in the GNN
        sphere_channels (int):      Number of spherical channels (one set per resolution)
        attn_hidden_channels (int): Number of hidden channels used during SO(2) graph attention
        num_heads (int):            Number of attention heads
        attn_alpha_head (int):      Number of channels for alpha vector in each attention head
        attn_value_head (int):      Number of channels for value vector in each attention head
        ffn_hidden_channels (int):  Number of hidden channels used during feedforward network
        lmax_list (int):            List of maximum degree of the spherical harmonics (1 to 10)
        mmax_list (int):            List of maximum order of the spherical harmonics (0 to lmax)
        edge_channels (int):        Number of channels for the edge invariant features

    Args recommended to be set by default:
        norm_type (str):            Type of normalization layer (['layer_norm', 'layer_norm_sh', 'rms_norm_sh'])
        grid_resolution (int):        Resolution of SO3_Grid
        num_sphere_samples (int):     Number of samples used to approximate the integration of the sphere in the output blocks
        use_atom_edge_embedding (bool):     Whether to use atomic embedding along with relative distance for edge scalar features
        share_atom_edge_embedding (bool):   Whether to share `atom_edge_embedding` across all blocks
        use_m_share_rad (bool):             Whether all m components within a type-L vector of one channel share radial function weights
        distance_function ("gaussian", "sigmoid", "linearsigmoid", "silu"):  Basis function used for distances
        attn_activation (str):      Type of activation function for SO(2) graph attention
        use_s2_act_attn (bool):     Whether to use attention after S2 activation. Otherwise, use the same attention as Equiformer
        use_attn_renorm (bool):     Whether to re-normalize attention weights
        ffn_activation (str):       Type of activation function for feedforward network
        use_gate_act (bool):        If `True`, use gate activation. Otherwise, use S2 activation
        use_grid_mlp (bool):        If `True`, use projecting to grids and performing MLPs for FFNs. 
        use_sep_s2_act (bool):      If `True`, use separable S2 activation when `use_gate_act` is False.
        alpha_drop (float):         Dropout rate for attention weights
        drop_path_rate (float):     Drop path rate
        proj_drop (float):          Dropout rate for outputs of attention and FFN in Transformer blocks
        weight_init (str):          ['normal', 'uniform'] initialization of weights of linear layers except those in radial functions
    """
    def __init__(
        self,
        input_types=[0, 1],
        input_channels=[3, 1],
        max_output_channels=3,
        max_neighbors=64,
        max_radius=1.0,
        num_layers=4,
        sphere_channels=128,
        attn_hidden_channels=128,
        num_heads=8,
        attn_alpha_channels=32,
        attn_value_channels=16,
        ffn_hidden_channels=512,
        edge_channels=128,
        lmax_list=[4],
        mmax_list=[3],

        norm_type='rms_norm_sh',
        grid_resolution=None, 
        num_sphere_samples=128,
        use_atom_edge_embedding=True, 
        share_atom_edge_embedding=False,
        use_m_share_rad=False,
        distance_function="gaussian",
        num_distance_basis=512, 

        attn_activation='scaled_silu',
        use_s2_act_attn=False, 
        use_attn_renorm=True,
        ffn_activation='scaled_silu',
        use_gate_act=False,
        use_grid_mlp=False, 
        use_sep_s2_act=True,

        alpha_drop=0.1,
        drop_path_rate=0.05, 
        proj_drop=0.0, 

        weight_init='normal'
    ):
        super().__init__()

        self.input_types = input_types
        self.input_channels = input_channels
        self.max_output_channels = max_output_channels

        self.max_neighbors = max_neighbors
        self.max_radius = max_radius
        self.cutoff = max_radius

        self.num_layers = num_layers
        self.sphere_channels = sphere_channels
        self.attn_hidden_channels = attn_hidden_channels
        self.num_heads = num_heads
        self.attn_alpha_channels = attn_alpha_channels
        self.attn_value_channels = attn_value_channels
        self.ffn_hidden_channels = ffn_hidden_channels
        self.norm_type = norm_type
        
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.grid_resolution = grid_resolution

        self.num_sphere_samples = num_sphere_samples

        self.edge_channels = edge_channels
        self.use_atom_edge_embedding = use_atom_edge_embedding 
        self.share_atom_edge_embedding = share_atom_edge_embedding
        if self.share_atom_edge_embedding:
            assert self.use_atom_edge_embedding
            self.block_use_atom_edge_embedding = False
        else:
            self.block_use_atom_edge_embedding = self.use_atom_edge_embedding
        self.use_m_share_rad = use_m_share_rad
        self.distance_function = distance_function
        self.num_distance_basis = num_distance_basis

        self.attn_activation = attn_activation
        self.use_s2_act_attn = use_s2_act_attn
        self.use_attn_renorm = use_attn_renorm
        self.ffn_activation = ffn_activation
        self.use_gate_act = use_gate_act
        self.use_grid_mlp = use_grid_mlp
        self.use_sep_s2_act = use_sep_s2_act
        
        self.alpha_drop = alpha_drop
        self.drop_path_rate = drop_path_rate
        self.proj_drop = proj_drop

        self.weight_init = weight_init
        assert self.weight_init in ['normal', 'uniform']

        self.device = 'cpu' #torch.cuda.current_device()

        self.grad_forces = False
        self.num_resolutions = len(self.lmax_list)
        
        # Weights for message initialization
        # self.sphere_embedding = nn.Embedding(self.max_num_elements, self.sphere_channels_all)
        assert 0 in self.input_types, "Input feature must have type-0 features!"
        assert self.input_types[-1] <= self.lmax_list[0], "Input feature type must be less than or equal to lmax"
        self.sphere_embedding = nn.Linear(self.input_channels[0], self.sphere_channels)
        
        # Initialize the function used to measure the distances between atoms
        assert self.distance_function in [
            'gaussian',
        ]
        if self.distance_function == 'gaussian':
            self.distance_expansion = GaussianSmearing(
                0.0,
                self.cutoff,
                600,
                2.0,
            )
            #self.distance_expansion = GaussianRadialBasisLayer(num_basis=self.num_distance_basis, cutoff=self.max_radius)
        else:
            raise ValueError
        
        # Initialize the sizes of radial functions (input channels and 2 hidden channels)
        self.edge_channels_list = [int(self.distance_expansion.num_output)] + [self.edge_channels] * 2

        # Initialize atom edge embedding
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            self.source_embedding = nn.Embedding(self.max_num_elements, self.edge_channels_list[-1])
            self.target_embedding = nn.Embedding(self.max_num_elements, self.edge_channels_list[-1])
            self.edge_channels_list[0] = self.edge_channels_list[0] + 2 * self.edge_channels_list[-1]
        else:
            self.source_embedding, self.target_embedding = None, None
        
        # Initialize the module that compute WignerD matrices and other values for spherical harmonic calculations
        self.SO3_rotation = nn.ModuleList()
        for i in range(self.num_resolutions):
            self.SO3_rotation.append(SO3_Rotation(self.lmax_list[i]))

        # Initialize conversion between degree l and order m layouts
        self.mappingReduced = CoefficientMappingModule(self.lmax_list, self.mmax_list)

        # Initialize the transformations between spherical and grid representations
        self.SO3_grid = ModuleListInfo('({}, {})'.format(max(self.lmax_list), max(self.lmax_list)))
        for l in range(max(self.lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(self.lmax_list) + 1):
                SO3_m_grid.append(
                    SO3_Grid(
                        l, 
                        m, 
                        resolution=self.grid_resolution, 
                        normalization='component'
                    )
                )
            self.SO3_grid.append(SO3_m_grid)

        # Edge-degree embedding
        self.edge_degree_embedding = EdgeDegreeEmbeddingPCD(
            self.sphere_channels,
            self.lmax_list,
            self.mmax_list,
            self.SO3_rotation,
            self.mappingReduced,
            self.input_channels[0],
            self.edge_channels_list,
            self.block_use_atom_edge_embedding,
            rescale_factor=self.max_neighbors / 2,  # TODO: Should be the degree of the graph, but here I just manually set it as the number of the neighbors
        )

        # Initialize the blocks for each layer of EquiformerV2
        self.blocks = nn.ModuleList()
        for i in range(self.num_layers):
            block = TransBlockV2(
                self.sphere_channels,
                self.attn_hidden_channels,
                self.num_heads,
                self.attn_alpha_channels,
                self.attn_value_channels,
                self.ffn_hidden_channels,
                self.sphere_channels, 
                self.lmax_list,
                self.mmax_list,
                self.SO3_rotation,
                self.mappingReduced,
                self.SO3_grid,
                self.input_channels[0],
                self.edge_channels_list,
                self.block_use_atom_edge_embedding,
                self.use_m_share_rad,
                self.attn_activation,
                self.use_s2_act_attn,
                self.use_attn_renorm,
                self.ffn_activation,
                self.use_gate_act,
                self.use_grid_mlp,
                self.use_sep_s2_act,
                self.norm_type,
                self.alpha_drop, 
                self.drop_path_rate,
                self.proj_drop
            )
            self.blocks.append(block)

        
        # Output blocks for energy and forces
        self.norm = get_normalization_layer(self.norm_type, lmax=max(self.lmax_list), num_channels=self.sphere_channels)
        self.output_head = SO2EquivariantGraphAttention(
                self.sphere_channels,
                self.attn_hidden_channels,
                self.num_heads, 
                self.attn_alpha_channels,
                self.attn_value_channels, 
                self.max_output_channels,
                self.lmax_list,
                self.mmax_list,
                self.SO3_rotation, 
                self.mappingReduced, 
                self.SO3_grid, 
                self.input_channels[0],
                self.edge_channels_list,
                self.block_use_atom_edge_embedding, 
                self.use_m_share_rad,
                self.attn_activation, 
                self.use_s2_act_attn, 
                self.use_attn_renorm,
                self.use_gate_act,
                self.use_sep_s2_act,
                alpha_drop=0.0
            )

        self.apply(self._init_weights)
        self.apply(self._uniform_init_rad_func_linear_weights)

        self.use_pbc = False
        self.otf_graph = True

    @torch.enable_grad()    # TODO: here I enable grad for the forwrad pass every time. Originally it has a condition for the force prediction
    def forward(self, data):
        ''' In GNNs, there is no batch axis. All graphs are concatenated into a single graph.
            So the first dim is the number of nodes in the concatenated graph.
        '''
        
        self.batch_size = data.batch[-1]
        self.dtype = data.pos.dtype
        self.device = data.pos.device

        num_pcds = len(data.pos)   # data.pcds is the concatenated point cloud data
        input_feature = data.feature  # should be in the shape of [num_pcds, feature_dim]
        inv_feature = input_feature[:, :self.input_channels[0]]

        (
            edge_index,
            edge_distance,
            edge_distance_vec,
            cell_offsets,
            _,  # cell offset distances
            neighbors,
        ) = self.generate_graph(data, use_pbc=False, otf_graph=True)

        ###############################################################
        # Initialize data structures
        ###############################################################

        # Compute 3x3 rotation matrix per edge
        edge_rot_mat = self._init_edge_rot_mat(
            data, edge_index, edge_distance_vec
        )

        # Initialize the WignerD matrices and other values for spherical harmonic calculations
        for i in range(self.num_resolutions):
            self.SO3_rotation[i].set_wigner(edge_rot_mat)

        ###############################################################
        # Initialize node embeddings
        ###############################################################

        # Init per node representations using an atomic number based embedding
        offset = 0
        x = SO3_Embedding(
            num_pcds,
            self.lmax_list,
            self.sphere_channels,
            self.device,
            self.dtype,
        )

        offset_res = 0
        offset = 0
        # Initialize the l = 0, m = 0 coefficients for each resolution
        for i in range(self.num_resolutions):
            if self.num_resolutions == 1:
                x.embedding[:, offset_res, :] = self.sphere_embedding(inv_feature)
            else:
                x.embedding[:, offset_res, :] = self.sphere_embedding(
                    inv_feature
                    )[:, offset : offset + self.sphere_channels]
            offset = offset + self.sphere_channels
            offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)

        # Edge encoding (distance and atom edge)
        edge_distance = self.distance_expansion(edge_distance)
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            # source_element = atomic_numbers[edge_index[0]]  # Source atom atomic number
            # target_element = atomic_numbers[edge_index[1]]  # Target atom atomic number
            # source_embedding = self.source_embedding(source_element)
            # target_embedding = self.target_embedding(target_element)
            source_embedding = self.source_embedding(inv_feature[edge_index[0]])
            target_embedding = self.target_embedding(inv_feature[edge_index[1]])
            edge_distance = torch.cat((edge_distance, source_embedding, target_embedding), dim=1)

        # Edge-degree embedding
        edge_degree = self.edge_degree_embedding(   # [num_pcds, (lmax+1)**2, edge_channels]
            inv_feature,
            edge_distance,
            edge_index)

        # replace part of the edge_degree embedding with the given input high order degree features
        input_feature_index = self.input_channels[0]
        for i in range(1, len(self.input_types)):
            # go to the type_i part of the edge_degree embedding
            current_begin_index = (self.input_types[i] - 1 + 1) ** 2
            current_type_i_length = 2*self.input_types[i] + 1

            edge_degree.embedding[:, current_begin_index:current_begin_index+current_type_i_length, :self.input_channels[i]] = \
                input_feature[:, input_feature_index:input_feature_index+current_type_i_length*self.input_channels[i]].reshape(-1, self.input_channels[i], current_type_i_length).transpose(1,2)    # [num_pcds, 2l+1, edge_channels]

            input_feature_index += current_type_i_length*self.input_channels[i]

        x.embedding = x.embedding + edge_degree.embedding

        ###############################################################
        # Update spherical node embeddings
        ###############################################################

        for i in range(self.num_layers):
            x = self.blocks[i](
                x,                  # SO3_Embedding
                inv_feature,
                edge_distance,
                edge_index,
                batch=data.batch    # for GraphDropPath
            )

        # Final layer norm
        x.embedding = self.norm(x.embedding)

        # Output Head. TODO: here I just mimic the original force predition implementation and choose to use a single SO2EquivariantGraphAttention layer as output head
        output = self.output_head(
            x,
            inv_feature,
            edge_distance,
            edge_index
        )
        
        # Modify this part to fit your predictions
        # Here I predict one type-0 feature and three type-1 features

        heatmap = output.embedding.narrow(1, 0, 1).squeeze(1)  # means extract the first dim, from 0, length 1. should be in the shape of [num_pcd, output_channel]
        heatmap = heatmap.mean(dim=1).unsqueeze(1)  # means average over the 2nd dim, and then unsqueeze to make it in the shape of [num_pcd, 1]
        # normalize with softmax
        start = 0
        for i in range(len(data.graph_pcd_num)):
            heatmap_i = heatmap[start:start+data.graph_pcd_num[i]]
            heatmap[start:start+data.graph_pcd_num[i]] = torch.nn.functional.softmax(heatmap_i, dim=1)
            start += data.graph_pcd_num[i]

        orientation = output.embedding.narrow(1, 1, 3).transpose(1,2)  # means extract the first dim, from 1, length 9. The original result's last dim is the channel, so we have to transpose it.
        orientation = orientation.reshape(-1, 9)  # reshape to [num_pcd, 9]
        for i in range(3):
            orientation[:, 3*i:3*(i+1)] /= (torch.norm(orientation[:, 3*i:3*(i+1)].clone(), dim=1).unsqueeze(1) + 1e-8)

        return heatmap, orientation

    # Initialize the edge rotation matrics
    def _init_edge_rot_mat(self, data, edge_index, edge_distance_vec):
        return init_edge_rot_mat(edge_distance_vec)
        

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


    def _init_weights(self, m):
        if (isinstance(m, torch.nn.Linear)
            or isinstance(m, SO3_LinearV2)
        ):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            if self.weight_init == 'normal':
                std = 1 / math.sqrt(m.in_features)
                torch.nn.init.normal_(m.weight, 0, std)

        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    
    def _uniform_init_rad_func_linear_weights(self, m):
        if (isinstance(m, RadialFunction)):
            m.apply(self._uniform_init_linear_weights)


    def _uniform_init_linear_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            std = 1 / math.sqrt(m.in_features)
            torch.nn.init.uniform_(m.weight, -std, std)

    
    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if (isinstance(module, torch.nn.Linear) 
                or isinstance(module, SO3_LinearV2)
                or isinstance(module, torch.nn.LayerNorm)
                or isinstance(module, EquivariantLayerNormArray)
                or isinstance(module, EquivariantLayerNormArraySphericalHarmonics)
                or isinstance(module, EquivariantRMSNormArraySphericalHarmonics)
                or isinstance(module, EquivariantRMSNormArraySphericalHarmonicsV2)
                or isinstance(module, GaussianRadialBasisLayer)):
                for parameter_name, _ in module.named_parameters():
                    if (isinstance(module, torch.nn.Linear)
                        or isinstance(module, SO3_LinearV2)
                    ):
                        if 'weight' in parameter_name:
                            continue
                    global_parameter_name = module_name + '.' + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)
        return set(no_wd_list)