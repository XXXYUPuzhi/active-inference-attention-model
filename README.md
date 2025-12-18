# Modeling Attention Shift via Active Inference & Drift Diffusion Model (DDM)

> **Current Status:** 🚧 Code refactoring and documentation in progress. Full Python implementation/simulation scripts will be released shortly.

##  Introduction
This project proposes a novel computational framework to explain **attention shifts** during sequential decision-making tasks. 

In traditional neuroscience, attention shifts are often viewed as random exploration. However, based on the **Active Inference** framework, I hypothesize that these shifts are regulatory mechanisms to maximize **information gain** and counteract **memory leakage** in the Prefrontal Cortex (PFC).

This model was originally proposed and developed during the **Computational Neuroscience Summer School (2025)** under the guidance of Prof. Ruben Moreno-Bote.

##  Theoretical Framework

The model integrates two powerful frameworks:
1.  **Drift Diffusion Model (DDM):** To model the decision accumulation process.
2.  **Active Inference:** To model the "drive" behind attention switches.

### Core Hypothesis
When an animal focuses on one option (e.g., Left), the internal representation of the unobserved option (e.g., Right) decays over time due to memory leakage. This decay increases the system's **entropy (uncertainty)**. When the entropy difference exceeds a certain threshold, the brain initiates an attention shift to "refresh" the working memory.

### Key Mathematical Formulation
We model the internal belief of an option's value as a Gaussian distribution that evolves over time:

$$\mathcal{P}_{X}(t) = \mathcal{N}(\mu_{X}, \sigma_{X}^{2}(t))$$

The uncertainty is quantified using **Shannon Entropy**:

$$H_{X}(t) = \frac{1}{2}\log(2\pi e \sigma_{X}^{2}(t))$$

The attention shift is triggered when the uncertainty exceeds the information gain threshold ($H_{th}$).

##  Key Features & "Cost of Attention"
A critical challenge in the initial model was that it predicted asymmetric psychometric curves in the late delay period, whereas experimental data showed **symmetry**. 

To resolve this, I introduced the **"Cost of Attention Shift"** variable. 
* The brain does not switch freely; it pays a cognitive/metabolic cost.
* By incorporating this cost into the objective function, the model successfully reproduces the symmetric behavioral features observed in monkey electrophysiology data.

##  Tech Stack & Roadmap
* **Languages:** Python (NumPy, SciPy), MATLAB
* **Modeling:** Active Inference, DDM, Bayesian Update
* **Status:**
    - [x] Theoretical derivation
    - [x] Mathematical formulation of Variance/Entropy updates
    - [x] Preliminary simulation (Summer School)
    - [ ] Code cleaning & public release (In Progress)

##  Reference
* Proposal Presentation: *Modeling attention shift in complex decision-making task* [Yu, 2025]
