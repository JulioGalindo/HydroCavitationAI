# State of the Art in Cavitation Detection in Hydraulic Turbines Using Neural Networks

## A) Introduction and Problem Description

Cavitation is a complex physical phenomenon occurring in liquid flow systems when the local static pressure drops below the liquid's vapor pressure at a constant temperature. This pressure drop leads to the formation of vapor-filled cavities or bubbles within the liquid. As these bubbles are transported by the flow into regions of higher pressure, they collapse violently. This implosion generates localized, high-intensity pressure waves (shockwaves) and micro-jets, which can cause significant problems in hydraulic machinery, particularly in hydraulic turbines [40, 41].

**The Problem of Cavitation in Hydraulic Turbines:**

Hydraulic turbines, such as Francis, Kaplan, and Pelton types, are critical components in hydroelectric power generation. They are designed to operate optimally at specific design points (Best Efficiency Point - BEP). However, modern grid demands, driven by the integration of intermittent renewable energy sources (solar, wind), often require turbines to operate under off-design conditions, including partial load, overload, and frequent start-stops [65, 66]. These off-design operations significantly increase the likelihood and severity of cavitation.

The detrimental effects of cavitation in hydraulic turbines are multifaceted:

1.  **Erosion Damage:** The repeated collapse of cavitation bubbles near the surfaces of turbine components (runner blades, guide vanes, draft tube) causes material pitting and erosion. This damage can range from minor surface roughening to significant material loss, leading to altered blade profiles and reduced structural integrity [8 (KD-CNN), 22, 40]. The damage mechanism involves high-speed liquid micro-jets and shockwaves impacting the material surface [40].
2.  **Performance Degradation:** Cavitation disrupts the flow patterns within the turbine, leading to a decrease in hydraulic efficiency, power output, and head [40, 41]. The formation of vapor cavities can block flow passages and alter the effective blade geometry.
3.  **Vibration and Noise:** The collapse of cavitation bubbles is a noisy process, generating broadband acoustic emissions and inducing vibrations in the turbine structure and connected systems [4, 22, 23]. These vibrations can lead to fatigue damage, loosening of components, and an overall reduction in the operational stability and lifespan of the machine.
4.  **Operational Instabilities:** Certain types of cavitation, such as draft tube swirl (vortex rope) at part load or overload conditions, can lead to severe pressure pulsations and power swings, threatening the stability of both the turbine and the electrical grid [65, 66].

**Need for Detection:**

Given the adverse effects, early and accurate detection of cavitation is crucial for:
* **Preventive Maintenance:** Identifying incipient cavitation allows for timely maintenance interventions, preventing extensive damage and costly repairs or replacements.
* **Operational Optimization:** Real-time cavitation monitoring can help operators adjust turbine operating points to avoid or minimize cavitation, extending the machine's life while optimizing power generation within safe limits.
* **Extended Operating Range:** With reliable detection, turbines might be operated closer to their cavitation limits, providing greater flexibility to the power grid, especially when balancing intermittent renewables [65].
* **Understanding Cavitation Dynamics:** Detailed detection and classification of cavitation states contribute to a better understanding of the phenomenon under various operating conditions, aiding in the design of more cavitation-resistant turbines.

Traditional methods for cavitation detection include visual inspection (often limited to model tests), performance drop analysis (which detects already developed cavitation), acoustic methods (listening for characteristic crackling sounds), and vibration analysis [4, 23, 40]. However, these methods can be subjective, late in detection, or struggle with complex noise and vibration signatures in operational environments.

**Neural Networks for Cavitation Detection:**

Neural networks (NNs), a subset of machine learning, have emerged as powerful tools for pattern recognition and classification in complex systems. Their ability to learn intricate relationships from data makes them well-suited for analyzing the complex, non-stationary, and noisy signals associated with cavitation (e.g., hydroacoustic emissions, vibrations, pressure pulsations) [4, 6, 8, 23].

This report explores the state of the art in utilizing neural networks for detecting cavitation in hydraulic turbines, focusing on two primary objectives:

1.  **Binary Detection (0 for no cavitation, 1 for cavitation):** This aims to provide a rapid, real-time indication of the presence or absence of cavitation, suitable for immediate operational feedback and alarm systems.
2.  **Multi-State Classification (e.g., 0 for no cavitation, up to 5 for severe cavitation, totaling 6 states):** This involves identifying not just the presence but also the intensity or type of cavitation, offering more granular information for diagnostics, prognostics, and refined operational adjustments.

The research delves into various signal processing techniques used to prepare input data for NNs, different NN architectures, and methods for enhancing model robustness and performance, including synthetic data generation.

## B) Study of Detection and Classification Methods

This section details prominent neural network-based methodologies and associated signal processing techniques for cavitation detection and classification in hydraulic turbines. For each method, we will discuss its principles, mathematical formulations where available, and provide conceptual Python snippets.

### 1. Singular Spectrum Analysis - Variational Mode Decomposition - Multi-Scale Convolutional Neural Network (SSA-VMD-MSCNN)

This hybrid approach combines advanced signal decomposition techniques (SSA and VMD) with a Multi-Scale Convolutional Neural Network (MSCNN) for diagnosing cavitation states in hydro turbines, particularly focusing on multi-state classification [4].

**a. Principles:**
The core idea is to first denoise and decompose the raw hydroacoustic signal into a set of Intrinsic Mode Functions (IMFs) that represent different oscillatory components. SSA is used as a pre-processing step to enhance the main signal components before VMD. VMD then adaptively decomposes the SSA-enhanced signal into a finite number of IMFs, each with a specific center frequency and bandwidth. These IMFs, which ideally separate different signal characteristics (including cavitation signatures and noise), are then fed as parallel input channels to an MSCNN. The MSCNN learns features from each IMF at multiple scales and then fuses these features for the final classification of the cavitation state.

**b. Mathematical Formulations:**

* **Singular Spectrum Analysis (SSA):**
    SSA is a non-parametric technique for time series analysis that decomposes a time series into a sum of identifiable components such as trend, oscillations, and noise. The main steps are [2, 3 (SSA tutorial), 54]:
    1.  **Embedding:** Convert the 1D time series $X = (x_1, ..., x_N)$ into a trajectory matrix $\mathbf{X}$ of dimension $L \times K$, where $L$ is the window length (embedding dimension) and $K = N - L + 1$.
        ```math
        \mathbf{X} = \begin{pmatrix}
        x_1 & x_2 & \cdots & x_K \\
        x_2 & x_3 & \cdots & x_{K+1} \\
        \vdots & \vdots & \ddots & \vdots \\
        x_L & x_{L+1} & \cdots & x_N
        \end{pmatrix}
        ```
        * $X$: Original 1D time series (scalar values).
        * $N$: Length of the time series (scalar, integer).
        * $L$: Window length or embedding dimension (scalar, integer). Chosen by the user, typically $2 \le L \le N/2$.
        * $K$: Number of columns in the trajectory matrix (scalar, integer).
        * $\mathbf{X}$: Trajectory matrix (matrix of scalars).

    2.  **Singular Value Decomposition (SVD):** Decompose the covariance matrix of the trajectory matrix, $\mathbf{S} = \mathbf{X}\mathbf{X}^T$. Let $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_L$ be the eigenvalues of $\mathbf{S}$, and $\mathbf{U}_1, \dots, \mathbf{U}_L$ be the corresponding eigenvectors (left singular vectors of $\mathbf{X}$). The principal components (PCs) $\mathbf{V}_i$ (right singular vectors) can be obtained. The SVD of $\mathbf{X}$ is $\mathbf{X} = \sum_{i=1}^{d} \sigma_i \mathbf{U}_i \mathbf{V}_i^T = \sum_{i=1}^{d} \mathbf{X}_i$, where $d = \text{rank}(\mathbf{X})$, $\sigma_i$ are the singular values ($\sqrt{\lambda_i}$), and $\mathbf{X}_i$ are elementary matrices.
        * $\mathbf{S}$: Covariance matrix of size $L \times L$ (matrix of scalars).
        * $\lambda_i$: Eigenvalues (scalar).
        * $\sigma_i$: Singular values (scalar).
        * $\mathbf{U}_i$: Left singular vectors of size $L \times 1$ (vector of scalars).
        * $\mathbf{V}_i$: Right singular vectors of size $K \times 1$ (vector of scalars).
        * $\mathbf{X}_i$: Elementary matrices of size $L \times K$ (matrix of scalars).

    3.  **Grouping:** Partition the set of indices $\{1, \dots, d\}$ into $m$ disjoint subsets $I_1, \dots, I_m$. This step is crucial for separating components. For denoising, one might group the eigentriples corresponding to the largest singular values that represent the signal and discard those related to noise.
        * $I_j$: Set of indices (integers).

    4.  **Diagonal Averaging (Reconstruction):** For each group $I_j$, reconstruct a component matrix $\mathbf{X}_{I_j} = \sum_{i \in I_j} \mathbf{X}_i$. Then, transform each $\mathbf{X}_{I_j}$ back into a time series $X^{(j)}$ of length $N$ by averaging along the anti-diagonals. The sum of these reconstructed series gives the decomposed components or the filtered series.
        * $X^{(j)}$: Reconstructed time series component (scalar values).

* **Variational Mode Decomposition (VMD):**
    VMD decomposes a signal $f(t)$ (scalar function of time $t$) into a discrete number of sub-signals (modes) $u_k(t)$ (scalar functions of time $t$) that have specific sparsity properties while reproducing the input signal. Each mode $u_k$ is assumed to be mostly compact around a center frequency $\omega_k$ (scalar). The goal is to minimize the sum of the estimated bandwidths of each mode, subject to the constraint that the modes sum up to the original signal. The constrained variational problem is [4 (VMD update steps), 51, 52 (PySDKit, VMD_python)]:
    ```math
    \min_{\{u_k\}, \{\omega_k\}} \left\{ \sum_k \left\| \partial_t \left[ \left( \delta(t) + \frac{j}{\pi t} \right) * u_k(t) \right] e^{-j\omega_k t} \right\|_2^2 \right\}
    ```
    subject to $\sum_k u_k(t) = f(t)$.
    * $f(t)$: Input signal (scalar function).
    * $u_k(t)$: $k$-th mode (scalar function).
    * $\omega_k$: Center frequency of the $k$-th mode (scalar).
    * $\partial_t$: Partial derivative with respect to time.
    * $\delta(t)$: Dirac delta function.
    * $j$: Imaginary unit.
    * $*$: Convolution operator.
    * $\|\cdot\|_2^2$: Squared $L_2$-norm.

    This is solved by introducing a quadratic penalty term and Lagrangian multipliers $\lambda(t)$ (scalar function):
    ```math
    \mathcal{L}(\{u_k\}, \{\omega_k\}, \lambda) = \alpha \sum_k \left\| \partial_t \left[ \left( \delta(t) + \frac{j}{\pi t} \right) * u_k(t) \right] e^{-j\omega_k t} \right\|_2^2 + \left\| f(t) - \sum_k u_k(t) \right\|_2^2 + \left\langle \lambda(t), f(t) - \sum_k u_k(t) \right\rangle
    ```
    where $\alpha$ (scalar) is the balancing parameter of the data-fidelity constraint. The solution is found iteratively using the Alternate Direction Method of Multipliers (ADMM), updating $u_k$, $\omega_k$, and $\lambda$ in the Fourier domain:
    Update for $u_k^{n+1}$ (where $n$ is the iteration number):
    ```math
    \hat{u}_k^{n+1}(\omega) = \frac{\hat{f}(\omega) - \sum_{i \neq k} \hat{u}_i^n(\omega) + \frac{\hat{\lambda}^n(\omega)}{2}}{1 + 2\alpha(\omega - \omega_k^n)^2}
    ```
    Update for $\omega_k^{n+1}$:
    ```math
    \omega_k^{n+1} = \frac{\int_0^\infty \omega |\hat{u}_k^{n+1}(\omega)|^2 d\omega}{\int_0^\infty |\hat{u}_k^{n+1}(\omega)|^2 d\omega}
    ```
    Update for $\lambda^{n+1}$:
    ```math
    \hat{\lambda}^{n+1}(\omega) = \hat{\lambda}^n(\omega) + \tau \left( \hat{f}(\omega) - \sum_k \hat{u}_k^{n+1}(\omega) \right)
    ```
    * $\hat{f}(\omega), \hat{u}_k(\omega), \hat{\lambda}(\omega)$: Fourier transforms of $f(t), u_k(t), \lambda(t)$ respectively (complex functions of frequency $\omega$).
    * $K$: Number of modes (scalar, integer).
    * $\tau$: Time-step of the dual ascent (scalar).

    In the SSA-VMD approach, an optimization algorithm (like a genetic algorithm or particle swarm optimization, not SSA itself for parameter tuning in this context, though the paper [4] refers to it as SSA-VMD suggesting SSA might be used for pre-filtering or feature selection for VMD parameter optimization) is used to optimize the selection of $K$ and $\alpha$ for VMD by defining a fitness function (e.g., based on permutation entropy (PE) of the IMFs) that the optimization algorithm attempts to optimize.

* **Multi-Scale Convolutional Neural Network (MSCNN):**
    The MSCNN architecture typically involves multiple parallel 1D convolutional pathways, each processing a different IMF (or a group of IMFs) [4]. Each pathway can have several convolutional and pooling layers to extract features at different scales from that specific IMF. The outputs of these pathways are then concatenated and fed into fully connected layers for final classification.
    A single channel (processing one IMF $u_k(t)$ which is a 1D vector of length $S_{len}$):
    Input IMF (shape: ($S_{len}$, 1))
    -> Conv1D (filters $F_1$, kernel size $KS_1$, stride $STR_1$) -> Activation (e.g., ReLU) -> MaxPool1D (pool size $PS_1$)
    -> Conv1D (filters $F_2$, kernel size $KS_2$, stride $STR_2$) -> Activation (e.g., ReLU) -> MaxPool1D (pool size $PS_2$)
    -> ... -> Flatten
    The concatenated flattened features from all IMF channels are then processed by Dense layers.
    * $F_i$: Number of filters in $i$-th convolutional layer (scalar, integer).
    * $KS_i$: Kernel size in $i$-th convolutional layer (scalar, integer).
    * $STR_i$: Stride in $i$-th convolutional layer (scalar, integer).
    * $PS_i$: Pool size in $i$-th pooling layer (scalar, integer).

**c. Python Snippets (Conceptual):**

* **VMD (using a library like `vmdpy` or `PyEMD`):**
    ```python
    import numpy as np
    # from vmdpy import VMD # Example library
    from PyEMD import VMD # Using PyEMD's VMD for this example

    # hydroacoustic_signal: 1D NumPy array of scalar values
    # K_optimal, alpha_optimal would be determined by an optimization algorithm (e.g., GA, PSO)
    # based on a fitness function (e.g., permutation entropy of IMFs)
    K_optimal = 5  # Example: Number of modes (scalar, integer)
    alpha_optimal = 2000  # Example: Penalty factor (scalar)
    tau = 0.  # Noise-slack (scalar, 0 means no noise slack)
    DC = False  # No DC part imposed (boolean)
    init = 1  # Initialize omegas uniformly (integer)
    tol = 1e-7 # Tolerance for convergence (scalar)

    # Assuming 'hydroacoustic_signal' is your 1D time series
    vmd_instance = VMD()
    # IMFs will be a NumPy array (matrix of scalars) where each row is an IMF
    IMFs = vmd_instance.vmd(hydroacoustic_signal, alpha=alpha_optimal, tau=tau, K=K_optimal, DC=DC, init=init, tol=tol)
    ```

* **SSA (conceptual, using a library like `pyts.decomposition`):**
    SSA implementation involves embedding, SVD, grouping, and diagonal averaging [2, 3 (SSA tutorial), 54].
    ```python
    import numpy as np
    from pyts.decomposition import SingularSpectrumAnalysis # Example library

    # X_signal: input time series (1D NumPy array of scalar values)
    # window_len: embedding window length (scalar, integer)
    window_len = 30 # Example
    
    # For denoising, one might select a subset of components to reconstruct the signal
    # n_components_to_keep = 5 # Example: keep first 5 components for signal, discard rest as noise
    # ssa_denoiser = SingularSpectrumAnalysis(window_size=window_len, groups=np.arange(n_components_to_keep))
    # X_denoised_2d = ssa_denoiser.fit_transform(X_signal.reshape(1, -1)) # pyts expects 2D array
    # X_denoised_1d = X_denoised_2d[0] # Get the reconstructed 1D signal

    # In SSA-VMD, the optimization of VMD parameters K and alpha
    # might use a fitness function. SSA itself is not directly optimizing VMD parameters in typical setups.
    # More likely, SSA is used for pre-filtering or feature extraction.
    # If SSA were used for optimization (e.g. as a search algorithm, which is less common for this):
    # def fitness_function_for_vmd_params(params_k_alpha, signal_for_vmd):
    #     K_val, alpha_val = int(params_k_alpha[0]), int(params_k_alpha[1])
    #     vmd_instance_opt = VMD()
    #     imfs_opt = vmd_instance_opt.vmd(signal_for_vmd, alpha=alpha_val, tau=0., K=K_val, DC=False, init=1, tol=1e-7)
    #     # Calculate a metric like sum of permutation entropies of IMFs
    #     # permutation_entropy_sum = calculate_permutation_entropy_sum(imfs_opt)
    #     # return permutation_entropy_sum # To be minimized
    # # Then an SSA-inspired optimization algorithm (or GA/PSO) would call this fitness function.
    ```

* **MSCNN (TensorFlow/Keras, conceptual for IMF channels):**
    The detailed MSCNN architecture would have multiple parallel channels, each processing an IMF [4].
    ```python
    import tensorflow as tf
    from tensorflow.keras.layers import Input, Conv1D, MaxPool1D, Flatten, Dense, Concatenate, BatchNormalization, Activation
    from tensorflow.keras.models import Model

    num_imfs = K_optimal  # From VMD (scalar, integer)
    # Assuming IMFs is a list/array of 1D signals (each an IMF)
    # seq_length = IMFs[0].shape[0] # Length of each IMF (scalar, integer)
    seq_length = 1024 # Example sequence length for each IMF (scalar, integer)
    num_classes = 6  # Example for 6 cavitation states (scalar, integer)

    input_channels = [] # List of Keras Input layers
    processed_channels = [] # List of Keras Tensors

    for i in range(num_imfs): 
        # Each IMF is a 1D vector, reshaped to (seq_length, 1) for Conv1D
        input_layer = Input(shape=(seq_length, 1), name=f'imf_input_{i+1}')
        input_channels.append(input_layer)
        
        # Convolutional architecture per channel, example inspired by [4]
        # This is a simplified single-scale path for one IMF channel for brevity
        # A true MSCNN would have multiple Conv1D with different kernel_sizes in parallel here
        
        # Path for current IMF
        x = Conv1D(filters=16, kernel_size=64, strides=1, padding='same')(input_layer)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPool1D(pool_size=2, strides=2)(x)
        
        x = Conv1D(filters=32, kernel_size=32, strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPool1D(pool_size=2, strides=2)(x)
        
        x = Flatten()(x)
        processed_channels.append(x)

    # Concatenate features from all IMF channels
    merged_features = Concatenate()(processed_channels)
    
    # Fully connected layers for classification
    dense1 = Dense(128)(merged_features) # Dense layer with 128 units (scalar, integer)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)
    
    output_layer = Dense(num_classes, activation='softmax')(dense1) # Output layer for num_classes

    mscnn_model = Model(inputs=input_channels, outputs=output_layer)
    mscnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # mscnn_model.summary() 
    ```

**d. Applicability:**
This method is particularly well-suited for multi-state detection due to its fine-grained signal decomposition and multi-scale feature learning. For real-time binary detection, the computational cost of the full SSA-VMD front-end could be a concern. However, the MSCNN itself, if fed with appropriately pre-processed features (perhaps from a faster VMD variant or selected IMFs), could be simplified for speed. The adaptive nature of VMD helps in handling non-stationary signals and varying noise conditions.

### 2. Double Hierarchical Residual Networks (DHRN) for Multi-Task Cavitation Analysis

This method focuses on the simultaneous detection of cavitation and recognition of its intensity using acoustic signals, particularly demonstrated for valves, with principles applicable to turbines [5, 6, 7].

**a. Principles and 1-D DHRN Architecture:**
The core of this technique is the use of 1-Dimensional Double Hierarchical Residual Networks (1-D DHRN) within a multi-task learning framework [5, 6]. A 1-D Double Hierarchical Residual Block (1-D DHRB) is constructed to capture sensitive features from acoustic signals in the frequency domain [5, 6, 7].

* **Residual Learning:** Based on ResNet [26 (He et al., 2016)], DHRN uses residual blocks to ease the training of very deep networks and enable better gradient flow. A standard residual block learns a residual mapping $\mathcal{F}(x)$ (scalar or vector function) such that the output is $H(x) = \mathcal{F}(x) + x$, where $x$ (scalar or vector) is the input to the block passed through a shortcut connection.
* **Hierarchical Structure:** The "Hierarchical" aspect in DHRB likely refers to the stacking of multiple residual units or the use of convolutional kernels of different sizes within a block to capture features at various scales or levels of abstraction. This is beneficial for complex acoustic signals where cavitation signatures might manifest across different frequency bands or with varying temporal complexities.
* **"Double" Hierarchy:** The term "Double" might imply two levels of hierarchical feature extraction. This could mean:
    1.  Inner hierarchy within each DHRB (e.g., multiple convolutional layers or parallel paths with different kernel sizes before the residual connection).
    2.  Outer hierarchy formed by stacking multiple DHRBs, where each block builds upon the features extracted by the previous ones.
    The goal is to effectively learn both local details and global contextual information from the frequency-domain representation of the acoustic signal.
* **1-D Operations:** Since the input is typically a 1D frequency spectrum (from FFT of a time window), the convolutions and other operations are 1-dimensional.
* **Multi-Task Learning:** The DHRN is trained to perform two related tasks simultaneously:
    1.  Binary cavitation detection (cavitation vs. no cavitation).
    2.  Multi-state cavitation intensity recognition (e.g., no cavitation, incipient, developing, severe).
    This is achieved by having a shared DHRN body for feature extraction, followed by two separate output heads (fully connected layers with appropriate activations like sigmoid for binary and softmax for multi-class) for each task. Multi-task learning can improve generalization by allowing the model to learn shared representations that are beneficial for all tasks.

**b. Data Augmentation with Sliding Window FFT (Swin-FFT):**
To address limited sample sizes, a data augmentation method called Sliding Window Fast Fourier Transform (Swin-FFT) is employed [5, 6, 7].
1.  A sliding window (defined by `window_size` and `step_size`) moves across the raw time-domain acoustic signal.
2.  For each segment captured by the window, an FFT is performed to transform it into the frequency domain.
3.  Each resulting frequency spectrum (or its magnitude) becomes an augmented training sample.
This directly generates more input samples in the frequency domain, which is the intended input for the 1-D DHRN.

**c. Mathematical Formulations:**

* **Standard Residual Block (1D):**
    Let $x_l$ (vector of scalars, representing features at layer $l$) be the input to the $l$-th residual block. The output $x_{l+1}$ (vector of scalars) is:
    ```math
    x_{l+1} = x_l + \mathcal{F}(x_l, \{W_i\}_l)
    ```
    where $\mathcal{F}(x_l, \{W_i\}_l)$ (vector function) represents the residual mapping learned by the weighted layers (e.g., two 1D convolutional layers with Batch Normalization and ReLU activation) in the block. For example:
    ```math
    \mathcal{F}(x_l) = W_2 * \text{ReLU}(BN(W_1 * x_l))
    ```
    where $*$ denotes 1D convolution, $W_1, W_2$ are filter weights (matrices of scalars, representing convolutional kernels), and BN is Batch Normalization. If dimensions of $x_l$ and $\mathcal{F}(x_l)$ differ (e.g., due to a change in the number of filters or downsampling via strides), a linear projection $W_s$ (matrix of scalars, typically implemented as a $1 \times 1$ convolution) is applied to $x_l$ in the shortcut connection: $x_{l+1} = W_s x_l + \mathcal{F}(x_l)$.

* **Fast Fourier Transform (FFT) for Swin-FFT:**
    For a windowed time-domain segment $s[n]$ (vector of $N_{FFT}$ scalar samples) of length $N_{FFT}$ (scalar, integer), its Discrete Fourier Transform (DFT), typically computed via FFT, is:
    ```math
    S[k] = \sum_{n=0}^{N_{FFT}-1} s[n] e^{-j \frac{2\pi}{N_{FFT}} kn}
    ```
    for $k = 0, 1, \dots, N_{FFT}-1$.
    * $s[n]$: Discrete time-domain signal samples in the window (scalar values).
    * $S[k]$: Discrete frequency-domain components (complex scalar values).
    * $N_{FFT}$: FFT length (scalar, integer).
    The input to DHRN is typically the magnitude spectrum $|S[k]|$ (vector of non-negative scalars).

* **Multi-Task Loss Function:**
    The overall loss $L_{total}$ (scalar) is a weighted sum of the losses for each task:
    ```math
    L_{total} = \lambda_{det} L_{detection} + \lambda_{int} L_{intensity}
    ```
    * $L_{detection}$: Binary cross-entropy for cavitation detection (scalar).
        ```math
        L_{detection} = - [y_{det} \log(\hat{y}_{det}) + (1-y_{det}) \log(1-\hat{y}_{det})]
        ```
    * $L_{intensity}$: Categorical cross-entropy for intensity recognition (scalar).
        ```math
        L_{intensity} = - \sum_{c=1}^{C} y_{int,c} \log(\hat{y}_{int,c})
        ```
    * $y_{det}, \hat{y}_{det}$: True and predicted binary labels (scalar, 0 or 1).
    * $y_{int,c}, \hat{y}_{int,c}$: True (one-hot encoded vector) and predicted probabilities (vector of scalars) for intensity class $c$. $C$ is the number of intensity states (scalar, integer).
    * $\lambda_{det}, \lambda_{int}$: Weighting factors for each task's loss (scalar).

**d. Python Snippets (Conceptual):**

* **1-D Residual Block (TensorFlow/Keras):**
    ```python
    import tensorflow as tf
    from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, Add, Dense, GlobalAveragePooling1D
    from tensorflow.keras.models import Model

    def residual_block_1d(input_tensor, filters, kernel_size=3, strides=1):
        # input_tensor: Keras tensor (batch_size, steps, input_filters)
        # filters: Number of output filters for convolutional layers (scalar, integer)
        # kernel_size: Size of the 1D convolution window (scalar, integer)
        # strides: Stride of the convolution (scalar, integer)

        # Main path
        x = Conv1D(filters, kernel_size, strides=strides, padding='same')(input_tensor)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv1D(filters, kernel_size, padding='same')(x) # Second convolution typically has no stride
        x = BatchNormalization()(x)

        # Shortcut path
        shortcut = input_tensor
        # If strides > 1 (downsampling) or number of filters changes, project shortcut
        if strides > 1 or input_tensor.shape[-1] != filters:
            shortcut = Conv1D(filters, 1, strides=strides, padding='same')(input_tensor) # 1x1 conv for projection
            shortcut = BatchNormalization()(shortcut)
        
        x = Add()([x, shortcut]) # Element-wise addition
        x = Activation('relu')(x)
        return x # Output tensor of shape (batch_size, new_steps, filters)

    # Example DHRN body (conceptual)
    # num_freq_bins = 512 # Example: number of frequency bins from FFT (scalar, integer)
    # input_layer = Input(shape=(num_freq_bins, 1)) # Shape: (frequency_bins, 1 channel)
    
    # Initial convolution (example)
    # x_feat = Conv1D(64, kernel_size=7, strides=2, padding='same')(input_layer) # 64 filters
    # x_feat = BatchNormalization()(x_feat)
    # x_feat = Activation('relu')(x_feat)
    # x_feat = tf.keras.layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(x_feat)

    # Stack of DHRBs (Double Hierarchical aspect would be in the internal DHRB design or stacking strategy)
    # For simplicity, showing stacked standard residual blocks
    # x_feat = residual_block_1d(x_feat, filters=64, kernel_size=3) 
    # x_feat = residual_block_1d(x_feat, filters=64, kernel_size=3)
    # x_feat = residual_block_1d(x_feat, filters=128, kernel_size=3, strides=2) # Downsample and increase filters
    # x_feat = residual_block_1d(x_feat, filters=128, kernel_size=3)
    # ... more blocks ...

    # Global Average Pooling before output heads
    # shared_features_pooled = GlobalAveragePooling1D()(x_feat) # Output shape: (batch_size, filters_of_last_block)

    # Output heads for multi-task learning
    # output_detection = Dense(1, activation='sigmoid', name='detection_output')(shared_features_pooled)
    # num_intensity_classes = 6 # Example: 0 to 5 (scalar, integer)
    # output_intensity = Dense(num_intensity_classes, activation='softmax', name='intensity_output')(shared_features_pooled)
    
    # dhrn_model = Model(inputs=input_layer, outputs=[output_detection, output_intensity])
    # losses_dict = {
    # 'detection_output': 'binary_crossentropy',
    # 'intensity_output': 'categorical_crossentropy'
    # }
    # loss_weights_dict = {'detection_output': 0.5, 'intensity_output': 0.5} # Example weights (scalars)
    # dhrn_model.compile(optimizer='adam', loss=losses_dict, loss_weights=loss_weights_dict, metrics=['accuracy'])
    # # dhrn_model.summary()
    ```

* **Swin-FFT Data Augmentation (Conceptual):**
    ```python
    import numpy as np
    from scipy.fft import fft

    def swin_fft_augmentation(time_series_signal, window_size, step_size, fft_length=None):
        # time_series_signal: 1D NumPy array of scalar values
        # window_size: Size of the sliding window (scalar, integer)
        # step_size: Step of the sliding window (scalar, integer)
        # fft_length: Length of the FFT, typically equal to window_size or padded (scalar, integer)
        
        if fft_length is None:
            fft_length = window_size
            
        augmented_spectra = [] # List to store spectra (vectors of non-negative scalars)
        num_frames = (len(time_series_signal) - window_size) // step_size + 1
        
        for i in range(num_frames):
            start_index = i * step_size
            end_index = start_index + window_size
            window = time_series_signal[start_index : end_index]
            
            # Apply FFT and take the magnitude of the first half (due to symmetry for real signals)
            spectrum = np.abs(fft(window, n=fft_length))[:fft_length // 2] 
            augmented_spectra.append(spectrum)
            
        return np.array(augmented_spectra) # Returns a matrix where each row is a spectrum
    ```

**e. Applicability:**
DHRN is highly suitable for the multi-state classification task (up to 6 states) due to its inherent intensity recognition capability. It also directly provides binary detection as one of its outputs. The use of frequency-domain signals and residual blocks makes it robust for analyzing acoustic emissions. The Swin-FFT augmentation is a practical way to increase dataset size. For real-time binary detection, the complexity of a very deep DHRN might need to be balanced against speed requirements.

### 3. Convolutional Neural Networks with Knowledge Distillation (KD-CNN)

This method aims to balance the accuracy of complex models with the computational efficiency required for real-time cavitation state identification in hydraulic turbines [8]. It leverages knowledge distillation, where a large, pre-trained "teacher" model transfers its knowledge to a smaller, faster "student" model.

**a. Principles of Knowledge Distillation:**
Knowledge Distillation (KD) is a model compression technique. The student model is trained to mimic the output distribution (soft labels) of the teacher model, in addition to learning from the true labels (hard labels) [9 (Hinton et al., 2015), 10, 11]. The soft labels from the teacher provide richer supervisory signals because they contain information about the teacher's "confidence" and how it relates different classes.

**b. Teacher and Student Model Architectures:**
The specific architectures used in [8] for acoustic emission signals are:
* **Teacher Model:** A 3-layer Convolutional Neural Network (CNN).
    * Input: Acoustic emission signal segment (e.g., 2000x1 vector).
    * Layers: Typically, each layer consists of a 1D convolution (e.g., 64 filters, kernel size 10, stride 1), followed by an activation function (e.g., ReLU), and a max-pooling layer (e.g., pool size 2, stride 2).
    * Output: A softmax layer for classification into $N_c$ cavitation states (e.g., 4 states: No Cavitation, Incipient, Developing, Critical).
* **Student Model:** A significantly simpler 1-layer CNN.
    * Input: Same as teacher (2000x1 vector).
    * Layers: One 1D convolutional layer (e.g., 2 filters, kernel size 10, stride 1), followed by activation and max-pooling.
    * Output: A softmax layer for classification into $N_c$ cavitation states.

The drastic reduction in layers and filters in the student model leads to significant speed-up.

**c. Distillation Process and Loss Function (with Temperature):**
The training process involves:
1.  Train the teacher model using the dataset with true (hard) labels until convergence.
2.  Use the trained teacher model to generate soft labels (probability distributions over classes) for the entire training set. These are obtained by applying a softmax function with a temperature parameter $T$ to the teacher's logits ($z_t$): $p_t = \text{softmax}(z_t/T)$. A $T > 1$ softens the probabilities, providing more information in the "dark knowledge" (relative probabilities of incorrect classes).
3.  Train the student model using a combined loss function:
    ```math
    L_{KD} = \alpha \cdot L_{CE\_hard}(y_{true}, \text{softmax}(z_s)) + (1-\alpha) \cdot L_{Distill}(p_t, \text{softmax}(z_s/T))
    ```
    * $L_{CE\_hard}$: Standard cross-entropy loss between true labels $y_{true}$ (vector of scalars, one-hot encoded) and student's predictions $\text{softmax}(z_s)$ (vector of scalars, probabilities).
        ```math
        L_{CE\_hard} = - \sum_i y_{true,i} \log(\text{softmax}(z_s)_i)
        ```
    * $L_{Distill}$: Distillation loss, often Kullback-Leibler (KL) divergence or cross-entropy between the teacher's soft labels $p_t$ and the student's softened predictions $\text{softmax}(z_s/T)$. If using cross-entropy:
        ```math
        L_{Distill} = - \sum_i p_{t,i} \log(\text{softmax}(z_s/T)_i) \cdot T^2
        ```
        The $T^2$ term is a scaling factor used in the original KD paper [9 (Hinton et al., 2015)] to ensure the magnitudes of gradients from the soft targets are roughly scaled to match those from the hard targets.
    * $z_s$: Logits from the student model (vector of scalars).
    * $\alpha$: Weighting factor (scalar, between 0 and 1) balancing the two loss terms.
    * $T$: Temperature parameter (scalar, typically $T \ge 1$).

**d. Performance and Real-Time Applicability:**
The KD-CNN approach in [8] identified cavitation states in under 2 seconds with over 97% accuracy for various working conditions. This demonstrates its suitability for real-time binary detection (by collapsing multi-state outputs if needed) and efficient multi-state classification.

**e. Python Snippets (Conceptual):**

* **Knowledge Distillation Loss Function (PyTorch, conceptual):**
    ```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def knowledge_distillation_loss(student_logits, teacher_logits, true_labels, temperature, alpha):
        # student_logits, teacher_logits: tensors of raw outputs (logits) from models
        # true_labels: tensor of ground truth labels (e.g., integer class indices for CrossEntropyLoss)
        # temperature: scalar, T >= 1
        # alpha: scalar, 0 <= alpha <= 1, weight for hard loss

        # Hard loss: Standard cross-entropy with true labels
        loss_hard = F.cross_entropy(student_logits, true_labels)

        # Soft loss: KL Divergence between softened teacher and student predictions
        # Softmax is applied with temperature to both teacher and student logits
        # KLDivLoss expects log-probabilities as input and probabilities as target.
        loss_soft = nn.KLDivLoss(reduction='batchmean')(
            F.log_softmax(student_logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1)  # Target is detached as it's not part of student's graph
        ) * (temperature * temperature) # Scaling by T^2

        # Combined loss
        total_loss = alpha * loss_hard + (1.0 - alpha) * loss_soft
        return total_loss

    # # --- Example Usage (Conceptual) ---
    # # Assuming teacher_model and student_model are defined PyTorch nn.Module instances
    # # And train_loader provides batches of (data, labels)

    # # teacher_model.eval() # Teacher is fixed during student training
    # # optimizer_student = torch.optim.Adam(student_model.parameters(), lr=0.001)
    # # T = 4.0 # Example temperature (scalar)
    # # alpha_kd = 0.3 # Example weight for hard loss (scalar)

    # # for data, labels in train_loader:
    # #     optimizer_student.zero_grad()
        
    # #     student_raw_outputs = student_model(data) # Logits from student
        
    # #     with torch.no_grad(): # Ensure teacher's gradients are not computed
    # #         teacher_raw_outputs = teacher_model(data) # Logits from teacher
            
    # #     loss = knowledge_distillation_loss(student_raw_outputs, teacher_raw_outputs, labels, T, alpha_kd)
    # #     loss.backward()
    # #     optimizer_student.step()
    ```

**f. Applicability:**
KD-CNN is excellent for deploying accurate models in resource-constrained environments or when real-time inference is paramount. For binary detection, a highly optimized student model can be very fast. For multi-state detection (e.g., 6 states), the teacher would be trained for these 6 states, and the student would learn to mimic this more complex output distribution. The student's architecture might need slight adjustments (e.g., more filters than the 2 used in the 4-state example) to handle the increased number of classes effectively while remaining efficient.

### 4. Auxiliary Classifier Generative Adversarial Networks (AC-GAN) for Robust Classification via Data Augmentation

AC-GANs are used to generate synthetic training data (specifically spectrograms) to enhance classifier robustness, especially when real data is limited or imbalanced across different cavitation states [15, 16].

**a. Principles of AC-GANs for Class-Conditional Data Generation:**
An AC-GAN extends the standard GAN framework. It consists of two neural networks:
* **Generator (G):** Takes a random noise vector $z$ (vector of scalars) and a class label $c$ (scalar or one-hot vector) as input and attempts to generate a data sample $X_{fake} = G(z, c)$ (e.g., a spectrogram) that looks like a real sample from class $c$.
* **Discriminator (D):** Takes a data sample $X$ (real or fake) as input and performs two tasks:
    1.  Determines if the sample is real or fake (source prediction).
    2.  Predicts the class label $c'$ of the sample (class prediction).

The key is that the generator must produce samples that are not only realistic but also identifiable as belonging to the specific input class $c$. This class-conditional generation is what makes AC-GANs suitable for augmenting datasets for supervised learning tasks.

**b. Generator and Discriminator Architectures:**
Architectures are typically based on Deep Convolutional GANs (DCGANs) when generating image-like data such as spectrograms [16].
* **Generator (G):**
    * Input: Concatenation or product of a noise vector $z$ (e.g., 100-dimensional, sampled from $N(0,1)$ or $U(-1,1)$) and an embedded class label $c_e$ (e.g., class index embedded into a dense vector).
    * Body: A series of transposed convolutional layers (sometimes called deconvolutional layers) to upsample the input into the desired spectrogram dimensions. Batch Normalization and ReLU/LeakyReLU activations are common.
    * Output: A layer with an activation like Tanh or Sigmoid to produce pixel values in the appropriate range for a spectrogram (e.g., normalized to [-1, 1] or [0, 1]).
* **Discriminator (D):**
    * Input: A spectrogram (real $X_{real}$ or fake $X_{fake}$).
    * Body: A series of convolutional layers to extract features. Strided convolutions are often used for downsampling instead of pooling layers. LeakyReLU activations are common.
    * Output Heads: After several convolutional layers, the feature maps are typically flattened and fed into two separate fully connected output branches:
        1.  **Source Prediction Head:** Outputs a single probability (via Sigmoid) indicating if the input is real (target 1) or fake (target 0).
        2.  **Class Prediction Head:** Outputs a probability distribution over the $N_c$ classes (via Softmax) indicating the predicted class of the input.

**c. Loss Functions for Adversarial Training and Classification:**
The training involves optimizing two loss functions simultaneously [15, 16]:
* **Discriminator Loss ($L_D$):** The discriminator aims to maximize its ability to distinguish real from fake and correctly classify real samples.
    ```math
    L_D = L_{source\_real} + L_{source\_fake} + L_{class\_real} 
    ```
    where:
    * $L_{source\_real} = -\mathbb{E}_{X_{real} \sim p_{data}}[\log D_{source}(X_{real})]$ (Discriminator wants $D_{source}(X_{real}) \to 1$)
    * $L_{source\_fake} = -\mathbb{E}_{z \sim p_z, c \sim p_c}[\log(1 - D_{source}(G(z,c)))]$ (Discriminator wants $D_{source}(G(z,c)) \to 0$)
    * $L_{class\_real} = -\mathbb{E}_{X_{real} \sim p_{data}, c_{real} \sim p_{data\_class}}[\log D_{class}(c_{real}|X_{real})]$ (Discriminator wants to correctly classify real samples)
    (Note: some formulations also include $L_{class\_fake}$ for the discriminator to classify fake samples, though the original AC-GAN paper focuses on classifying real samples correctly with $D_{class}$ and having $G$ produce classifiable samples.)

* **Generator Loss ($L_G$):** The generator aims to produce samples that the discriminator thinks are real and that are correctly classifiable as the intended class $c$.
    ```math
    L_G = L_{source\_gen} + L_{class\_gen}
    ```
    where:
    * $L_{source\_gen} = -\mathbb{E}_{z \sim p_z, c \sim p_c}[\log D_{source}(G(z,c))]$ (Generator wants $D_{source}(G(z,c)) \to 1$)
    * $L_{class\_gen} = -\mathbb{E}_{z \sim p_z, c \sim p_c}[\log D_{class}(c|G(z,c))]$ (Generator wants $D_{class}$ to correctly classify its generated samples according to the input condition $c$)

The terms $D_{source}(X)$ and $D_{class}(c|X)$ represent the outputs of the source prediction head and class prediction head of the discriminator, respectively. $p_z$ is the noise distribution, $p_c$ is the distribution of class labels (e.g., uniform if balancing), $p_{data}$ is the real data distribution, and $p_{data\_class}$ is the true class label for real data.

**d. Generating Synthetic Spectrograms and Augmenting Training Data:**
Once the AC-GAN is trained, the generator $G$ can be used to create new synthetic spectrograms by providing it with random noise vectors $z$ and desired class labels $c$. These synthetic spectrograms can then be added to the real training dataset to augment it. This is particularly useful for:
* **Increasing dataset size:** Especially for classes with few real samples.
* **Balancing classes:** Generating more samples for minority classes.
* **Improving classifier robustness:** Exposing the subsequent classifier (e.g., a ResNet or DHRN) to a wider variety of data, potentially covering gaps in the real data distribution.
The study [17] (CovidGAN) showed an increase in CNN classification accuracy from 85% to 95% by augmenting with GAN-generated images. Similar benefits are expected for cavitation spectrograms.

**e. Python Snippets (Conceptual):**
* **AC-GAN Training Loop (PyTorch, conceptual):**
    ```python
    import torch
    import torch.nn as nn
    import torch.optim as optim

    # Assume Generator_AC and Discriminator_AC models are defined
    # netG = Generator_AC(nz, ngf, nc, num_classes, embedding_dim) # nz: noise_dim, ngf: gen_feat_maps, nc: num_channels_img, num_classes
    # netD = Discriminator_AC(nc, ndf, num_classes) # ndf: disc_feat_maps

    # Loss functions
    criterion_adv = nn.BCELoss()  # For real/fake source
    criterion_aux = nn.NLLLoss() # For class labels (if D_class outputs log_softmax) or CrossEntropyLoss

    # Optimizers
    # optimizerD = optim.Adam(netD.parameters(), lr=lr_d, betas=(beta1, 0.999))
    # optimizerG = optim.Adam(netG.parameters(), lr=lr_g, betas=(beta1, 0.999))

    # Fixed noise and labels for visualizing G's progress
    # fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device) 
    # fixed_class_labels_for_G = torch.randint(0, num_classes, (batch_size,), device=device)

    # for epoch in range(num_epochs):
    #     for i, data_batch in enumerate(dataloader_real):
    #         real_samples = data_batch[0].to(device) # Spectrograms
    #         real_class_labels = data_batch[1].to(device) # True class labels for real samples
    #         batch_size_current = real_samples.size(0)

    #         # --- Train Discriminator ---
    #         netD.zero_grad()
    #         # Real samples
    #         label_real_source = torch.full((batch_size_current,), 1.0, device=device) # Target for real is 1
    #         output_source_real, output_class_real = netD(real_samples)
    #         errD_real_adv = criterion_adv(output_source_real.squeeze(), label_real_source)
    #         errD_real_aux = criterion_aux(output_class_real, real_class_labels)
            
    #         # Fake samples
    #         noise_input = torch.randn(batch_size_current, nz, device=device) # Noise vector z
    #         # Class labels for G to generate (can be random or targeted)
    #         gen_class_labels = torch.randint(0, num_classes, (batch_size_current,), device=device) 
    #         fake_samples = netG(noise_input, gen_class_labels) # G generates fake samples
            
    #         label_fake_source = torch.full((batch_size_current,), 0.0, device=device) # Target for fake is 0
    #         output_source_fake, output_class_fake = netD(fake_samples.detach()) # Detach to avoid G grads
    #         errD_fake_adv = criterion_adv(output_source_fake.squeeze(), label_fake_source)
    #         # Optionally, D can also be penalized for misclassifying fake samples' classes
    #         # errD_fake_aux = criterion_aux(output_class_fake, gen_class_labels) 
            
    #         errD = errD_real_adv + errD_real_aux + errD_fake_adv # + errD_fake_aux
    #         errD.backward()
    #         # optimizerD.step()

    #         # --- Train Generator ---
    #         netG.zero_grad()
    #         # We want D to think fake samples are real
    #         output_source_fake_for_G, output_class_fake_for_G = netD(fake_samples) 
    #         errG_adv = criterion_adv(output_source_fake_for_G.squeeze(), label_real_source) # Try to fool D's source prediction
    #         errG_aux = criterion_aux(output_class_fake_for_G, gen_class_labels) # G wants D to classify correctly
            
    #         errG = errG_adv + errG_aux
    #         errG.backward()
    #         # optimizerG.step()
    ```

**f. Applicability:**
AC-GANs are powerful for data augmentation, especially for multi-state classification where obtaining sufficient real data for each of the 6 states can be challenging. The generated spectrograms can then be used to train any of the other classifier models (MSCNN, DHRN, KD-CNN, standard ResNets). This technique is primarily an offline data enhancement step rather than a real-time detection method itself.

### 5. General Hydroacoustic Signal-Based Neural Network Approaches

This category encompasses a broader range of methods that utilize hydroacoustic signals (from hydrophones or Acoustic Emission sensors) as input to various neural network architectures, often with a focus on specific signal processing steps.

**a. Signal Sources and Characteristics:**
Hydroacoustic signals are direct measurements of pressure waves propagating through the water or structure-borne elastic waves.
* **Acoustic Emission (AE) Sensors:** These sensors are typically piezoelectric and capture high-frequency elastic waves (often in the ultrasonic range, e.g., 100 kHz - 1 MHz) generated by the rapid release of energy from cavitation bubble collapses or material micro-fractures [22, 23, 67 (Fernández-Osete et al., 2024)]. They are sensitive to transient events and can be mounted externally on the turbine casing or components.
* **Hydrophones:** These are underwater microphones that capture pressure fluctuations in the fluid, including the noise radiated by cavitation. They can cover a broad frequency range.
* **Vibration Sensors (Accelerometers):** While not strictly "hydroacoustic," accelerometers mounted on the turbine structure capture vibrations induced by cavitation and other hydraulic phenomena [23, 68 (Valentín et al., 2019)]. Their signals are often analyzed in conjunction with acoustic signals.

The choice of sensor influences the frequency content and characteristics of the signal. AE signals are typically broadband and transient, while hydrophone signals might contain more continuous flow noise in addition to cavitation signatures.

**b. Signal Processing and Feature Extraction:**
Raw hydroacoustic signals are often non-stationary and contaminated by noise from machinery operation, flow turbulence, and electromagnetic interference (EMI) [20, 21, 36, 37, 61, 62]. Effective pre-processing is crucial:
1.  **Filtering:** Band-pass filtering is common to isolate frequency ranges where cavitation is most prominent and to remove out-of-band noise [23, 67].
2.  **Time-Frequency Transformation:**
    * **Short-Time Fourier Transform (STFT):** Widely used to generate spectrograms (time vs. frequency vs. amplitude/power) which serve as image-like inputs to CNNs [15, 16, 23, 25, 69 (Piczak, DCASE2017)].
        ```math
        X(t, f) = \sum_{n=-\infty}^{\infty} x[n] w[n-t] e^{-j2\pi fn}
        ```
        * $x[n]$: Discrete time signal (scalar values).
        * $w[n]$: Window function (e.g., Hamming, Hann) (scalar values).
        * $X(t,f)$: STFT coefficient at time $t$ and frequency $f$ (complex scalar).
        The magnitude squared $|X(t,f)|^2$ or log-magnitude $20 \log_{10} |X(t,f)|$ is typically used.
    * **Mel-Spectrograms:** STFT spectrograms are often converted to the Mel scale to mimic human auditory perception, which can improve performance in acoustic tasks [25, 70 (Vafeiadis et al., DCASE2017), 71 (Schindler et al., DCASE2017)].
    * **Wavelet Transform (e.g., Continuous Wavelet Transform - CWT, Wavelet Packet Decomposition - WPD):** Offers variable time-frequency resolution, which can be advantageous for capturing both transient bursts and longer-duration features of cavitation [21, 24, 72 (Qian et al., DCASE2017), 73 (Dong et al., 2019)]. WPD decomposes the signal into various frequency sub-bands, and energy coefficients from these bands can be used as features [73].
        The CWT of a signal $x(t)$ is:
        ```math
        CWT(a,b) = \frac{1}{\sqrt{a}} \int_{-\infty}^{\infty} x(t) \psi^*\left(\frac{t-b}{a}\right) dt
        ```
        * $\psi(t)$: Mother wavelet function (scalar function).
        * $a$: Scale parameter (scalar, $a>0$).
        * $b$: Translation parameter (scalar).
        Scalograms are the time-scale representations derived from CWT.
3.  **Signal Decomposition:**
    * **Empirical Mode Decomposition (EMD) / Ensemble EMD (EEMD):** Adaptive methods to decompose non-stationary signals into IMFs. EEMD improves upon EMD by adding white noise to alleviate mode mixing [21, 74 (Dao et al.)]. The IMFs can then be analyzed individually or used to reconstruct a denoised signal.
4.  **Statistical Features:** From time-domain signals or their decomposed components, statistical features like RMS, peak-to-peak, kurtosis, skewness, crest factor, shape factor, etc., can be extracted [1, 75 (Sha et al., 2022 - XGBoost)].
5.  **Envelope Analysis:** Using the Hilbert transform to obtain the envelope of high-frequency signals, which can then be analyzed in the frequency domain (demodulation) to reveal low-frequency modulations indicative of cavitation dynamics (e.g., shedding frequencies) [23, 67, 76 (Feldman, 2011)].
    The analytic signal $z(t)$ of a real signal $x(t)$ is:
    ```math
    z(t) = x(t) + j \mathcal{H}\{x(t)\}
    ```
    where $\mathcal{H}\{x(t)\}$ is the Hilbert transform of $x(t)$:
    ```math
    \mathcal{H}\{x(t)\} = \frac{1}{\pi} \text{p.v.} \int_{-\infty}^{\infty} \frac{x(\tau)}{t-\tau} d\tau
    ```
    The envelope is $A(t) = |z(t)|$.

**c. Neural Network Architectures:**
* **Convolutional Neural Networks (CNNs):** The most common choice when using spectrograms as input. Architectures vary from simple (few layers) [25, 70] to deeper ones like ResNet variants [23, 26]. 1D CNNs are used for time-series or frequency spectra directly [4, 5, 6, 8].
* **Recurrent Neural Networks (RNNs), LSTMs, GRUs:** Suitable for capturing temporal dependencies in sequences of features or directly from time-series data [74]. Often combined with CNNs (CNN-LSTM architectures) where CNNs extract local features and RNNs model temporal sequences of these features [74].
* **Autoencoders (AE):** Used for unsupervised feature learning or anomaly detection. An AE trained on "no cavitation" data can detect cavitation when the reconstruction error for new data is high [29, 63 (Zhao et al., 2020 - Pump Turbines)].
* **Radial Basis Function (RBF) Networks:** Used in [73] after WPD and PCA for cavitation status identification in centrifugal pumps.
* **Multilayer Perceptrons (MLPs):** Often used as classifiers on top of handcrafted or learned features [75].

**d. Challenges and Considerations:**
* **Noise Robustness:** A primary challenge. Techniques like EEMD denoising [74] or domain adaptation [23] are employed.
* **Sensor Position and Type:** The characteristics of the acquired signal and thus the optimal NN approach can depend heavily on the sensor type (AE, hydrophone, accelerometer) and its mounting location [49, 68, 77 (Schmidt et al., 2014 - Kaplan)].
* **Computational Cost for Real-Time:** Complex pre-processing and deep NNs can be computationally intensive.
* **Data Scarcity:** Addressed by data augmentation techniques (see Section D).
* **Domain Shift:** Models trained on one machine/condition may not generalize well to others [23].

**e. Python Snippets (Conceptual):**

* **EEMD Denoising (using `PyEMD`):**
    ```python
    import numpy as np
    from PyEMD import EEMD

    # signal_1d: 1D NumPy array of scalar values
    # eemd = EEMD()
    # IMFs = eemd(signal_1d) # Decomposes into IMFs

    # Denoising often involves selecting a subset of IMFs (e.g., lower frequency ones)
    # and summing them back to reconstruct a cleaner signal.
    # Thresholding or other criteria might be used to select IMFs.
    # For example, if first 'k_denoise' IMFs are considered signal and rest noise:
    # denoised_signal = np.sum(IMFs[:k_denoise, :], axis=0) 
    ```

* **Spectrogram Generation (using `librosa`):**
    ```python
    import librosa
    import numpy as np

    # y: audio time series (1D NumPy array of scalar values)
    # sr: sampling rate (scalar, integer)
    # n_fft = 2048 # FFT window size (scalar, integer)
    # hop_length = 512 # Hop length for STFT (scalar, integer)
    # n_mels = 128 # Number of Mel bands (scalar, integer)

    # Mel-spectrogram
    # S_mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    # S_mel_db = librosa.power_to_db(S_mel, ref=np.max) # Convert to dB scale
    # S_mel_db would be the input to a 2D CNN, shape: (n_mels, num_frames)
    ```

**f. Applicability:**
General hydroacoustic approaches are versatile. For binary detection, simpler CNNs on well-chosen features (e.g., specific frequency bands from spectrograms or statistical features) can be effective and fast. For multi-state classification (e.g., 6 states), deeper architectures (ResNets, CNN-LSTMs) and more sophisticated feature extraction (e.g., from VMD/EEMD IMFs, or detailed Mel-spectrograms) are generally required to capture the nuances between different cavitation intensities. The choice of pre-processing (e.g., EEMD vs VMD) and NN architecture should be guided by the specific signal characteristics and computational constraints.

### 6. Multi-Index Fusion Adaptive VMD for Cavitation Feature Extraction

This approach, detailed in [20] and [27], focuses on enhancing the Variational Mode Decomposition (VMD) process by adaptively determining the number of decomposition layers ($K$) and the penalty factor ($\alpha$). The goal is to improve the extraction of cavitation features by better handling noise and preserving crucial cavitation information from hydroacoustic signals.

**a. Principles:**
Standard VMD requires $K$ and $\alpha$ to be pre-set. An inappropriate choice can lead to over-decomposition (generating spurious modes or excessive noise) or under-decomposition (losing important signal details). This method proposes:
1.  **Adaptive Parameter Selection for VMD:**
    * **Mean Filter Envelope Extremum Method:** Used for a preliminary analysis of the noisy signal to determine the number of dominant frequency points. This helps in setting a reasonable initial search range for $K$ and $\alpha$ [27, 80 (ssrn-5199304)].
    * **Sparrow Search Algorithm (SSA) Optimization:** An intelligent optimization algorithm (inspired by sparrow foraging behavior) is used to globally optimize $K$ and $\alpha$. The fitness function for SSA combines envelope entropy (to prevent over-decomposition) and signal energy loss ratio (to prevent under-decomposition and loss of cavitation energy) [27, 80].
        ```math
        \text{Fitness} = \gamma \cdot E_p + \delta_{loss}
        ```
        * $E_p$: Envelope entropy (scalar), measures signal sparsity. Lower $E_p$ suggests regularity.
        * $\delta_{loss}$: Signal energy loss ratio (scalar), measures how much energy is lost/retained after VMD.
        * $\gamma$: Weighting coefficient (scalar).
2.  **IMF Screening:** After VMD decomposition with optimized parameters, the resulting IMFs are screened. Envelope entropy is calculated for each IMF, and irregular noise components (those with high entropy) are removed or filtered out to improve the signal-to-noise ratio of the features fed to the classifier [27, 80].
3.  **Multi-Resolution S-Transform (MST):** The selected/denoised IMF components are then subjected to MST to generate time-frequency feature maps. MST is an improvement over the standard S-Transform, allowing for different time-frequency resolutions in different frequency bands, which is beneficial for non-stationary cavitation signals [27, 80].
4.  **Deep Learning for Classification:** The multi-scale time-frequency feature maps from MST are fed into an improved Inception-ResNet model (MIR-CBAM) for classification. This model incorporates asymmetric convolution kernels and the Convolutional Block Attention Module (CBAM) to enhance multi-scale feature extraction and focus on critical disturbance information [27, 80].

**b. Mathematical Formulations:**

* **Mean Filter Envelope Extremum Method:**
    This involves FFT of the noisy signal, normalization, iterative mean filtering of the spectrum, local maximum detection on the smoothed spectrum envelope, and thresholding to identify dominant frequency points ($M_2$) [27, 80]. These $M_2$ points guide the search range for VMD's $K$ and $\alpha$.

* **Sparrow Search Algorithm (SSA) for VMD Parameter Optimization:**
    SSA is a population-based heuristic algorithm. It involves "Producers" (sparrows searching for food) and "Scroungers" (sparrows following producers). Positions (representing $K, \alpha$ pairs) are updated based on fitness values (from the composite envelope entropy and energy loss function). A "Watcher" mechanism prevents premature convergence [27, 80]. The update equations for producer positions $X_{i,j}$ are:
    ```math
    X_{i,j}^{t+1} = \begin{cases} X_{i,j}^t \cdot \exp(-i / (\beta \cdot iter_{max})) & \text{if } R_2 < ST \\ X_{i,j}^t + Q \cdot L & \text{if } R_2 \ge ST \end{cases}
    ```
    * $X_{i,j}^t$: Position of $i$-th sparrow in $j$-th dimension at iteration $t$ (scalar).
    * $\beta$: Random number in $(0,1]$ (scalar).
    * $iter_{max}$: Maximum iterations (scalar, integer).
    * $R_2 \in [0,1]$: Alarm value (scalar).
    * $ST \in [0.5,1]$: Safety threshold (scalar).
    * $Q$: Random number from normal distribution (scalar).
    * $L$: $1 \times D$ matrix of ones (vector of scalars).
    (Refer to [27, 80] for Scrounger and Watcher update rules.)

* **Envelope Entropy ($E_p$) of an IMF:**
    Given an IMF $u(t)$, its envelope $a(t)$ is obtained (e.g., via Hilbert transform). The probability distribution $p_j$ of envelope values is:
    ```math
    p_j = \frac{a_j}{\sum_{i=1}^{N_{env}} a_i}
    ```
    where $a_j$ are discrete samples of the envelope $a(t)$, and $N_{env}$ is the number of samples.
    ```math
    E_p = - \sum_{j=1}^{N_{env}} p_j \log_2 p_j
    ```
    * $a_j$: $j$-th sample of the envelope (scalar).
    * $p_j$: Normalized envelope value (scalar).

* **Multi-Resolution S-Transform (MST):**
    The standard S-Transform is:
    ```math
    S(\tau, f) = \int_{-\infty}^{\infty} x(t) \frac{|f|}{\sqrt{2\pi}} e^{-\frac{(\tau-t)^2 f^2}{2}} e^{-j2\pi ft} dt
    ```
    MST modifies the Gaussian window's standard deviation $\sigma(f) = 1/|f|$ to be frequency-dependent with more flexibility, often using different parameters for low and high-frequency regions [27, 80]:
    ```math
    \sigma(f) = \frac{d}{a+b|f|^c} \quad \text{or piecewise definitions for } \sigma_1(f), \sigma_2(f)
    ```
    * $a, b, c, d$: Tuning parameters for the window (scalars).

* **Improved Inception-ResNet with CBAM (MIR-CBAM):**
    This architecture combines Inception modules (for multi-scale features), ResNet blocks (for deep training), and CBAM attention mechanisms (channel and spatial attention) to focus on salient features in the MST-derived time-frequency maps [27, 80].
    * **CBAM - Channel Attention Module (CAM):**
        ```math
        M_c(F) = \sigma(MLP(AvgPool(F)) + MLP(MaxPool(F)))
        ```
        * $F$: Input feature map (tensor).
        * $M_c(F)$: Channel attention map (vector of scalars).
        * $\sigma$: Sigmoid function.
        * $MLP$: Multi-Layer Perceptron.
    * **CBAM - Spatial Attention Module (SAM):**
        ```math
        M_s(F') = \sigma(Conv^{7\times7}([AvgPool(F'); MaxPool(F')]))
        ```
        * $F'$: Channel-refined feature map (tensor).
        * $M_s(F')$: Spatial attention map (matrix of scalars).
        * $Conv^{7\times7}$: A convolutional layer with a $7 \times 7$ filter.

**c. Python Snippets (Conceptual):**

* **Sparrow Search Algorithm (Conceptual Outline):**
    ```python
    # import numpy as np

    # def fitness_vmd(params_k_alpha, signal):
    #     K, alpha = int(params_k_alpha[0]), int(params_k_alpha[1])
    #     # Run VMD with K, alpha
    #     # imfs = run_vmd(signal, K, alpha)
    #     # Calculate envelope_entropy_sum = sum(calculate_envelope_entropy(imf) for imf in imfs)
    #     # Calculate energy_loss_ratio = calculate_energy_loss(signal, imfs)
    #     # gamma = 0.5 # Example weight
    #     # return gamma * envelope_entropy_sum + energy_loss_ratio

    # # SSA parameters
    # # population_size = 20
    # # max_iterations = 50
    # # dim = 2 # For K and alpha
    # # lower_bounds = [K_min, alpha_min] # From Mean Filter Envelope Extremum
    # # upper_bounds = [K_max, alpha_max]

    # # Initialize sparrow population (positions for K, alpha)
    # # sparrows = np.random.uniform(low=lower_bounds, high=upper_bounds, size=(population_size, dim))
    
    # # Main SSA loop
    # # for t in range(max_iterations):
    # #     Calculate fitness for all sparrows using fitness_vmd
    # #     Sort sparrows by fitness
    # #     Update positions of producers (best sparrows)
    # #     Update positions of scroungers (other sparrows following producers)
    # #     Apply watcher mechanism (random jumps for some sparrows)
    # #     Ensure positions stay within bounds
    
    # # best_params = sparrows[0] # Sparrow with the best fitness
    # # K_optimal_ssa, alpha_optimal_ssa = int(best_params[0]), int(best_params[1])
    ```

* **Improved Inception-ResNet with CBAM (TensorFlow/Keras, very high-level):**
    ```python
    # import tensorflow as tf
    # from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add
    # from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Multiply, Reshape, Permute
    # from tensorflow.keras.models import Model

    # def cbam_block(input_feature, ratio=8):
    #     # Channel Attention Module
    #     channel_avg_pool = GlobalAveragePooling2D()(input_feature)
    #     channel_max_pool = tf.keras.layers.GlobalMaxPooling2D()(input_feature) # Corrected
    #     # Shared MLP
    #     # mlp_shared_1 = Dense(input_feature.shape[-1] // ratio, activation='relu')
    #     # mlp_shared_2 = Dense(input_feature.shape[-1])
    #     # avg_out = mlp_shared_2(mlp_shared_1(channel_avg_pool))
    #     # max_out = mlp_shared_2(mlp_shared_1(channel_max_pool))
    #     # channel_attention = Activation('sigmoid')(Add()([avg_out, max_out]))
    #     # channel_attention = Reshape((1, 1, input_feature.shape[-1]))(channel_attention)
    #     # feature_after_channel_attention = Multiply()([input_feature, channel_attention])
        
    #     # Spatial Attention Module (simplified)
    #     # kernel_size_spatial = 7
    #     # spatial_avg_pool = tf.reduce_mean(feature_after_channel_attention, axis=[3], keepdims=True)
    #     # spatial_max_pool = tf.reduce_max(feature_after_channel_attention, axis=[3], keepdims=True)
    #     # spatial_concat = tf.keras.layers.Concatenate(axis=3)([spatial_avg_pool, spatial_max_pool])
    #     # spatial_attention = Conv2D(1, kernel_size_spatial, padding='same', activation='sigmoid')(spatial_concat)
    #     # refined_feature = Multiply()([feature_after_channel_attention, spatial_attention])
    #     # return refined_feature
    #     pass # Placeholder for brevity

    # # Inception-ResNet block (conceptual)
    # # def inception_resnet_block_with_cbam(input_tensor, ...):
    # #     # ... Inception module structure ...
    # #     # ... Residual connection ...
    # #     # output_block = cbam_block(output_residual)
    # #     # return output_block
    #     pass # Placeholder

    # # input_mst_maps = Input(shape=(height, width, num_imf_channels)) # MST maps as input
    # # x = ... # Stem layers (initial convolutions)
    # # x = inception_resnet_block_with_cbam(x, ...)
    # # ... more blocks ...
    # # x = GlobalAveragePooling2D()(x)
    # # output_classification = Dense(num_classes, activation='softmax')(x)
    # # model_mir_cbam = Model(inputs=input_mst_maps, outputs=output_classification)
    ```

**d. Applicability:**
This method is designed for high accuracy in noisy environments and for distinguishing subtle differences between cavitation states, making it suitable for multi-state classification (e.g., 6 states). The adaptive VMD and MST provide rich feature maps. The complexity of the SSA optimization and the MIR-CBAM model might make it challenging for very fast real-time binary detection unless significant optimizations are made or simpler versions are used. Its main strength lies in robust feature extraction for accurate classification.

---
*The next sections will cover: C) Comparative Evaluation, D) Synthetic Data Generation (consolidating AC-GAN and other methods from the survey [32, 33]), E) Conclusions, and F) Full Bibliography.*
