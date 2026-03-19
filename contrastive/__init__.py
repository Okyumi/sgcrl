"""Contrastive RL agent."""

from contrastive.agents import DistributedContrastive
from contrastive.builder import ContrastiveBuilder
from contrastive.config import ContrastiveConfig
from contrastive.config import target_entropy_from_env_spec
from contrastive.learning import ContrastiveLearner
from contrastive.networks import apply_policy_and_sample
from contrastive.networks import ContrastiveNetworks
from contrastive.networks import make_networks

# Continual RL extensions
from contrastive.continual_config import ContinualConfig
from contrastive.continual_config import CONTINUAL_TASK_SEQUENCE
from contrastive.continual_learning import ContinualContrastiveLearner
from contrastive.continual_builder import ContinualContrastiveBuilder
from contrastive.knowledge_pool import KnowledgePool
