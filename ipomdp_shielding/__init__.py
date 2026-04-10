"""
IMDP Belief Propagation Library

A library for studying different methods for belief propagation
in interval partially observable Markov decision processes.

Modules:
- Models: Core data structures (MDP, IMDP, IPOMDP, POMDP, Confidence)
- Propagators: Belief propagation algorithms (ExactHMM, MinMaxHMM, LFP)
- Evaluation: Shield construction and evaluation
- CaseStudies: Example applications (Taxinet)
"""

from . import Models
from . import Propagators
from . import Evaluation
from . import CaseStudies

__all__ = ['Models', 'Propagators', 'Evaluation', 'CaseStudies']
__version__ = '0.1.0'
