"""
Constants for metric learning models and their default parameters.
Based on Table 2 from SISAP25 paper.
"""

import sys
import os

# Add implementations folder to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'implementations'))

from implementations.OASIS import OASIS
from implementations.OMDML import OMDML
from implementations.SORS import SORS, AdaSORS
from implementations.Robust import RobustODML
from implementations.AROMA import FactoredAROMA
from implementations.OPML import OPML
from implementations.MLOML import MLOML
from implementations.POLA import POLA


# Default parameters for each model and feedback type based on Table 2 from the paper
# Format: model_name -> feedback_type -> parameters
# Feedback types correspond to Feedback Mechanisms 1, 2, 3 from Section 4.2 of the paper:
#   1 = Random Pairing (single positive + single negative, repeated)
#   2 = Single Negative & All Positives (one negative paired with all positives)
#   3 = All Possible Pairs (all negatives paired with all positives)
MODEL_FEEDBACK_DEFAULTS = {
    'OASIS': {
        1: {'C': 7e-3, 'enforce_psd': False},  # Mechanism 1: Random Pairing
        2: {'C': 1e-3, 'enforce_psd': False},  # Mechanism 2: Single Negative & All Positives
        3: {'C': 1e-3, 'enforce_psd': False},  # Mechanism 3: All Possible Pairs
    },
    'OMDML': {
        1: {'beta': 7e-2, 'C': 1e-2, 'gamma': 1e-3},  # Mechanism 1
        2: {'beta': 1e-2, 'C': 2e-2, 'gamma': 7e-2},  # Mechanism 2
        3: {'beta': 1e-3, 'C': 4e-3, 'gamma': 7e-3},  # Mechanism 3 (same as Mechanism 2)
    },
    'SORS': {
        1: {'eta': 4e-3, 'reg_lambda': 4e-6, 'reg_type': 'offdiag'},  # Mechanism 1
        2: {'eta': 1e-3, 'reg_lambda': 1e-6, 'reg_type': 'offdiag'},  # Mechanism 2
        3: {'eta': 1e-3, 'reg_lambda': 1e-6, 'reg_type': 'offdiag'},  # Mechanism 3 (same as Mechanism 2)
    },
    'AdaSORS': {
        1: {'eta': 1e-3, 'reg_lambda': 4e-4, 'reg_type': 'offdiag', 'delta': 4e-6},  # Mechanism 1
        2: {'eta': 7e-5, 'reg_lambda': 4e-4, 'reg_type': 'offdiag', 'delta': 4e-4},  # Mechanism 2
        3: {'eta': 7e-4, 'reg_lambda': 4e-4, 'reg_type': 'offdiag', 'delta': 1e-5},  # Mechanism 3
    },
    'RobustODML': {
        1: {'C': 7e-4, 'eta': 1.0, 'max_hq_iter': 20},  # Mechanism 1
        2: {'C': 7e-4, 'eta': 7e-2, 'max_hq_iter': 1},  # Mechanism 2
        3: {'C': 7e-4, 'eta': 7e-2, 'max_hq_iter': 1},  # Mechanism 3 (same as Mechanism 2)
    },
    'FactoredAROMA': {
        1: {'r': 700.0},  # Mechanism 1
        2: {'r': 700.0},  # Mechanism 2
        3: {'r': 700.0},  # Mechanism 3 (same for all)
    },
    'OPML': {
        1: {'gamma': 7e-3},  # Mechanism 1
        2: {'gamma': 4e-4},  # Mechanism 2
        3: {'gamma': 4e-4},  # Mechanism 3 (same as Mechanism 2)
    },
    'MLOML': {
        1: {'n_layers': 5, 'gamma': 4e-3, 'activation': 'tanh', 'lambda_reg': 1e-3},  # Mechanism 1
        2: {'n_layers': 3, 'gamma': 4e-4, 'activation': 'tanh', 'lambda_reg': 7e-4},  # Mechanism 2
        3: {'n_layers': 2, 'gamma': 4e-4, 'activation': 'tanh', 'lambda_reg': 7e-6},  # Mechanism 3
    },
    'POLA': {
        1: {'b_init': 4e-2, 'gamma': 7e-1},  # Mechanism 1 only (Mechanisms 2 and 3 are n/a in Table 2)
    },
}

# Model configurations
METRIC_LEARNING_MODELS = {
    'OASIS': {'class': OASIS, 'pair_type': 'triplets'},
    'OMDML': {'class': OMDML, 'pair_type': 'pairs'},
    'SORS': {'class': SORS, 'pair_type': 'triplets'},
    'AdaSORS': {'class': AdaSORS, 'pair_type': 'triplets'},
    'RobustODML': {'class': RobustODML, 'pair_type': 'pairs'},
    'FactoredAROMA': {'class': FactoredAROMA, 'pair_type': 'pairs'},
    'OPML': {'class': OPML, 'pair_type': 'triplets'},
    'MLOML': {'class': MLOML, 'pair_type': 'pairs'},
    'POLA': {'class': POLA, 'pair_type': 'pairs'},
}

# One-sentence model descriptions from the paper's related work (Section 2.2)
MODEL_DESCRIPTIONS = {
    'OASIS': 'OASIS learns a bilinear similarity via an online passive-aggressive algorithm.',
    'OMDML': 'OMDML extends online metric learning to multi-modal data by separately optimising each modality\'s Mahalanobis distance metric and then fusing them via the Hedge algorithm.',
    'SORS': 'SORS is a sparse online relative similarity learning algorithm that reduces memory and computational costs by imposing an off-diagonal ℓ₁ norm on the similarity matrix.',
    'AdaSORS': 'AdaSORS enhances SORS by incorporating adaptive second-order regularisation via an AdaGrad update for improved performance.',
    'RobustODML': 'Robust-ODML replaces the standard hinge loss with a rescaled hinge loss and learns a low-rank Mahalanobis distance metric via adaptive weighting, robustly mitigating the effects of outliers and label noise.',
    'FactoredAROMA': 'AROMA leverages adaptive regularisation to learn a distribution over matrix models by simultaneously estimating a weight matrix and its associated confidence via a covariance matrix; the factored variant captures inter-feature correlations.',
    'OPML': 'OPML leverages a one-pass triplet sampling strategy and learns a square transformation matrix via a closed-form solution at O(d²) cost, avoiding explicit PSD constraints.',
    'MLOML': 'MLOML utilises a multilayer framework that progressively builds multiple hierarchical metric spaces (based on Mahalanobis distance metric) through integrated nonlinear transformations.',
    'POLA': 'POLA introduces an explicit threshold, updated via successive projections alongside the Mahalanobis distance metric, to enforce a finite margin between similarly and differently labelled examples.',
}

# Full citations from the paper's references section
MODEL_CITATIONS = {
    'OASIS': 'G. Chechik, U. Shalit, V. Sharma, and S. Bengio. An online algorithm for large scale image similarity learning. In Advances in Neural Information Processing Systems, volume 22, pages 306–314, 2009.',
    'OMDML': 'P. Wu, S. C. H. Hoi, P. Zhao, C. Miao, and Z. Liu. Online multi-modal distance metric learning with application to image retrieval. IEEE Transactions on Knowledge and Data Engineering, 28(2):454–467, 2016.',
    'SORS': 'D. Yao, P. Zhao, C. Yu, H. Jin, and B. Li. Sparse online relative similarity learning. In 2015 IEEE International Conference on Data Mining, pages 529–538, 2015.',
    'AdaSORS': 'D. Yao, P. Zhao, C. Yu, H. Jin, and B. Li. Sparse online relative similarity learning. In 2015 IEEE International Conference on Data Mining, pages 529–538, 2015.',
    'RobustODML': 'D. Zabihzadeh, A. Tuama, A. Karami-Mollaee, and S. J. Mousavirad. Low-rank robust online distance/similarity learning based on the rescaled hinge loss. Applied Intelligence, 53(1):634–657, 2023.',
    'FactoredAROMA': 'K. Crammer and G. Chechik. Adaptive regularization for weight matrices. arXiv preprint arXiv:1206.4639, 2012.',
    'OPML': 'W. Li, Y. Gao, L. Wang, L. Zhou, J. Huo, and Y. Shi. OPML: A one-pass closed-form solution for online metric learning. Pattern Recognition, 75:302–314, 2018.',
    'MLOML': 'W. Li, Y. Liu, J. Huo, Y. Shi, Y. Gao, L. Wang, and J. Luo. A multilayer framework for online metric learning. IEEE Transactions on Neural Networks and Learning Systems, 34(10):6701–6713, 2023.',
    'POLA': 'S. Shalev-Shwartz, Y. Singer, and A. Y. Ng. Online and batch learning of pseudo-metrics. In Proceedings of the 21st International Conference on Machine Learning (ICML) 2004, volume 69, 2004.',
}

# Parameter descriptions for each model (from Section 4.3 of the paper)
MODEL_PARAM_DESCRIPTIONS = {
    'OASIS': {
        'C': 'Sets the aggressiveness of the update. Higher the parameter -> higher personalization effect, but also higher scaling factor.',
        'enforce_psd': 'Forces the learned matrix to be positive semi-definite. Low impact on the personalization.',
    },
    'OMDML': {
        'beta': 'Sets the discount weight in the Hedge update. Low impact on the personalization.',
        'C': 'The regularisation parameter. Higher the parameter -> higher personalization effect, but also higher scaling factor.',
        'gamma': 'Sets the margin for separation. Low impact on the personalization.',
    },
    'SORS': {
        'eta': 'The learning rate. Higher the parameter -> higher personalization effect, but also higher scaling factor.',
        'reg_lambda': 'Regularisation parameter for controlling sparsity. Higher the parameter -> lower personalization effect, but also lower scaling factor.',
        'reg_type': 'Selects full L1 (\'l1\') or off-diagonal L1 regularisation (\'offdiag\'). Lower impact on the personalization.',
    },
    'AdaSORS': {
        'eta': 'The learning rate. Higher the parameter -> higher personalization effect, but also higher scaling factor.',
        'reg_lambda': 'Regularisation parameter for controlling sparsity. Higher the parameter -> lower personalization effect, but also lower scaling factor.',
        'reg_type': 'Selects full L1 (\'l1\') or off-diagonal L1 regularisation (\'offdiag\'). Lower impact on the personalization.',
        'delta': 'The smoothness parameter. Higher the parameter -> lower personalization effect, but also lower scaling factor.',
    },
    'RobustODML': {
        'C': 'Sets the aggressiveness of the update. Higher the parameter -> higher personalization effect, but also higher scaling factor.',
        'eta': 'Controls the robustness of the loss. Higher the parameter -> lower personalization effect, but also lower scaling factor.',
        'max_hq_iter': 'Maximum number of iterations in the Half-Quadratic algorithm. Higher the parameter -> higher personalization effect, higher scaling factor, but also higher learning time.',
    },
    'FactoredAROMA': {
        'r': 'The regularisation parameter. Higher the parameter -> lower personalization effect, but also lower scaling factor.',
    },
    'OPML': {
        'gamma': 'The regularisation parameter. Higher the parameter -> higher personalization effect, but also higher scaling factor.',
    },
    'MLOML': {
        'n_layers': 'Sets the number of metric layers. Higher the parameter -> higher personalization effect, higher scaling factor, but also higher learning time.',
        'gamma': 'The regularisation parameter. Higher the parameter -> higher personalization effect, but also higher scaling factor.',
        'activation': 'Sets the nonlinearity function (\'relu\', \'sigmoid\', or \'tanh\'). Sigmoid and Relu results in general in lower scaling factors than Tanh. All functions improve personalization.',
        'lambda_reg': 'Used for the additional loss layer. Low impact on the personalization.',
    },
    'POLA': {
        'b_init': 'The initial threshold. Higher the parameter -> lower personalization effect, lower scaling factor.',
        'gamma': 'The relaxation parameter for the inseparable case. Higher the parameter -> lower personalization effect, lower scaling factor.',
    },
}

# Default model and feedback type
DEFAULT_MODEL = 'OMDML'
DEFAULT_FEEDBACK_TYPE = 2  # Mechanism 2

FEEDBACK_TYPE_NAMES = {
    1: 'Random Pairing',
    2: 'Single Negative & All Positives',
    3: 'All Possible Pairs',
}

FEEDBACK_TYPE_DESCRIPTIONS = {
    1: 'A single positive image and a single negative image are randomly selected to form one triplet for metric learning of the matrix. This process repeats for a configurable number of iterations. Sampling can be done with or without replacement.',
    2: 'A single negative image is chosen randomly and paired with all positive images, creating multiple triplets per iteration. This is repeated for each negative image (without replacement). Triplets can be processed as a batch or individually.',
    3: 'All negative images are paired with all positive images, generating the complete set of triplets for comprehensive metric learning of the matrix. The full set is fed to the model as a single batch.',
}

# Configurable parameters for each feedback type, with defaults, descriptions, and allowed values
FEEDBACK_TYPE_PARAMS = {
    1: {
        'num_iterations': {
            'default': 10,
            'type': 'int',
            'description': 'Number of triplets to generate for metric learning. Higher the number, the better personalization effect, but with higher scaling factor.',
            'allowed_values': [1, 2, 4, 8, 16, 32, 64],
            'min': 1,
            'max': 64,
        },
        'with_replacement': {
            'default': True,
            'type': 'bool',
            'description': 'Whether to sample with replacement. If False, the same image will not be re-selected in multiple iterations.',
            'allowed_values': [True, False],
        },
    },
    2: {
        'batch_mode': {
            'default': True,
            'type': 'bool',
            'description': 'If True, all triplets for a given negative are processed as a batch. If False, each triplet is processed individually (one at a time).',
            'allowed_values': [True, False],
        },
    },
    3: {
        # Feedback Type 3 has no additional user-configurable parameters.
        # All negatives are paired with all positives and processed as a single batch.
    },
}
