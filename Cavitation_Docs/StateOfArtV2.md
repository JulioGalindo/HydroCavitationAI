# State of the Art in Cavitation Detection in Hydraulic Turbines Using Neural Networks

## Introduction

Cavitation is the formation and collapse of vapor bubbles in a liquid flow, which occurs when local pressure drops below the vapor pressure.  In hydraulic turbines, cavitation can cause serious damage (erosion, noise, vibration) and efficiency loss. Early detection of cavitation is therefore critical for the safe and efficient operation of hydropower equipment. Traditional cavitation detection methods rely on signal processing and hand-crafted features (e.g. spectral energy or kurtosis of acoustic emission) , but these often fail under complex operating conditions. Recent advances use deep learning to automatically detect and classify cavitation from vibration or acoustic signals. In particular, convolutional neural networks (CNNs) have been applied to acoustic spectrograms to detect cavitation presence or intensity. This document surveys the literature on neural-network-based cavitation detection in hydraulic turbines, covering architectures, signal processing, data augmentation, and performance analysis.

Cavitation detection can be formulated as a binary problem (cavitation vs. no cavitation) or multiclass (e.g. incipient, developed, severe states). Common sensors include hydrophones for acoustic emission and accelerometers for vibration; some recent works also consider electromagnetic or pressure signals. The signals are typically nonstationary and noisy, requiring advanced time-frequency processing such as Variational Mode Decomposition (VMD) or wavelet transforms. Data scarcity is a challenge, so synthetic data generation and augmentation (e.g. via GANs) are employed.

This review covers the following topics:

* **Binary cavitation detection with SSA-VMD and multiscale CNNs**: Methods combining Singular Spectrum Analysis and VMD to preprocess hydroacoustic signals, followed by a multiscale CNN (MSCNN).
* **State classification using SSA-VMD and CNNs**: Extending SSA-VMD preprocessing for multiclass cavitation state identification.
* **1-D DHRN multitask networks**: Using 1-D dual-hierarchical residual networks for simultaneous cavitation detection and intensity estimation from valve acoustics.
* **Data augmentation and 1-D DHRN for valve acoustics**: The “Swin-FFT” sliding-window FFT augmentation and the structure of the 1-D DHRN used for valve signal analysis.
* **Knowledge Distillation CNN (KD-CNN)**: A teacher-student CNN approach (3-layer teacher, 1-layer student) for fast cavitation state recognition with high accuracy.
* **AC-GAN architectures**: Auxiliary-Classifier GANs for data augmentation and robust classification of cavitation from spectrograms.
* **CNN with spectrogram inputs**: Standard CNNs trained on acoustic spectrograms (with preprocessing) enhanced by GAN-generated data.
* **Performance analysis**: Comparative metrics of accuracy, inference speed, and robustness for the above classifiers, including discussions of domain adaptation.
* **Synthetic data generation**: Methods for generating synthetic cavitation signals (GAN-based, VAEs, etc.), with mathematical formulation and examples (illustrative spectrograms if available).
* **Conclusions**: Key findings and recommendations for real-time cavitation detection systems.

Throughout, we present mathematical details (denoting vectors, matrices explicitly) in LaTeX `math` blocks and cite relevant literature at the beginning of each paragraph as `【{source}†L..】`. Equations define all symbols. Existing results are integrated from English and Spanish sources where applicable. This review aims to guide engineers and researchers in implementing real-time binary and multiclass cavitation detection using neural networks.

## A) Problem Description and Background

Cavitation in hydraulic turbines typically occurs under off-design or low-load conditions, where pressure drops.  The cavitation process generates broadband noise (from a few kHz up to MHz) and high-frequency shock emissions. These acoustic emissions can be captured by hydrophones or piezoelectric sensors. Detecting cavitation via acoustic signals is challenging due to noise from turbulence and mechanical parts, as well as the nonstationarity of the signal.

A classical approach is to compute features such as the energy in specific bands or the kurtosis of the acoustic signal. Modern methods use neural networks to learn features from raw or preprocessed signals. For example, Look *et al.* (2018) proposed training a CNN directly on ultrasonic-range spectrograms of hydrophone data. Harsch *et al.* (2023) (Gaisser et al.) addressed the domain shift problem by pre-processing and domain-adversarial training so that a single CNN can generalize across different turbine types. Knowledge distillation techniques compress complex models into simpler ones for speed. Auxiliary tasks (e.g. intensity estimation) and multi-modal signals (acoustic + vibration + electromagnetic) have also been explored for robustness.

**Signal processing:** Key pre-processing tools include time-frequency transforms and decomposition methods. For instance, Short-Time Fourier Transform (STFT) produces spectrograms, often followed by *dynamic range compression* (DCR) to emphasize weak cavitation signals:

```math
f(x) = \log(1 + C x),
```

with parameter \$C\$. Filter banks (e.g. Mel or triangular windows) are applied to reduce spectrogram dimensionality.

Another approach is Variational Mode Decomposition (VMD), which decomposes a 1-D signal \$x(t)\$ into \$K\$ intrinsic mode functions \$u\_k(t)\$ centered at frequencies \$\omega\_k\$. The VMD optimization is formulated as:

```math
\{u_k,\omega_k\} = \arg\min \sum_{k=1}^K \Big\| \partial_t [u_k(t)e^{-j\omega_k t}] \Big\|_2^2 
\quad\text{subject to}\quad \sum_{k=1}^K u_k(t) = x(t).
```

This constrains the sum of modes to equal the input. VMD has a solid variational basis and can separate overlapping components. In practice, one also uses an augmented Lagrangian approach (e.g. ADMM) to solve this optimization. The resulting modes \$u\_k\$ are narrowband; their instantaneous frequencies and envelopes are used as features.

Singular Spectrum Analysis (SSA) is another decomposition method for nonstationary signals.  Given a time series \$x\_1,\dots,x\_N\$, SSA constructs the Hankel trajectory matrix \$X\in\mathbb{R}^{L\times K}\$ (\$K=N-L+1\$) by

```math
X = \begin{bmatrix}
x_1 & x_2 & \dots & x_{K} \\
x_2 & x_3 & \dots & x_{K+1} \\
\vdots & & \ddots & \vdots \\
x_{L} & x_{L+1} & \dots & x_{N}
\end{bmatrix}.
```

One then computes the singular value decomposition \$X = U \Sigma V^\top\$.  Each rank-1 component \$X\_m = \sigma\_m u\_m v\_m^\top\$ corresponds to a principal component of the signal. Averaging along the anti-diagonals of \$X\_m\$ yields a reconstructed time series \$x^{(m)}(t)\$. In the SSA-VMD method, SSA serves to separate slow trends and oscillatory components before applying VMD.

In summary, cavitation produces distinct acoustic signatures that can be enhanced via signal decomposition and transformation. The next sections review neural network models built on such processed signals.

## B) Detection and Classification Methods

### B.1 Binary Cavitation Detection with SSA-VMD-MSCNN

**Method overview:** Li *et al.* (2024) proposed an SSA-VMD-MSCNN model for **binary** cavitation detection from hydroacoustic signals. The input is raw acoustic time series sampled by an underwater sensor. The method has three stages (Fig.1):

* **SSA-VMD Layer:** The time series is first processed by singular spectrum analysis (SSA) to remove noise and capture dominant oscillatory modes. SSA outputs a smoothed trend \$x\_{\text{trend}}(t)\$ and fluctuations \$x\_{\text{fluc}}(t)\$. The fluctuations are then decomposed by Variational Mode Decomposition (VMD) into \$K\$ modes \${u\_k(t)}\$. The number of modes \$K\$ is chosen by an optimization step (e.g. using Sparrow Search Algorithm) that maximizes an index combining energy, entropy, and correlation. This yields adaptive mode decomposition tailored to cavitation signals.

* **Feature Extraction:** Each VMD mode \$u\_k(t)\$ is band-limited around a center frequency \$\omega\_k\$. Li *et al.* treat each mode as a channel. They also compute reconstructed signals for pairs of modes and compute their energy or entropy as additional features. All resulting signals (the modes or groups of modes) form a multichannel input matrix \$X \in \mathbb{R}^{K \times T}\$ (where \$T\$ is time length).

* **Multiscale CNN (MSCNN):** This channelized signal \$X\$ is fed into a CNN with multiple filter sizes to capture both fine and coarse features. Typically, convolution kernels of several widths (e.g. 3,5,7) are applied in parallel (like Inception or multi-scale CNN) and concatenated. The MSCNN extracts features hierarchically, followed by fully connected layers and a softmax to output the binary cavitation label.

Li *et al.* report that their SSA-VMD-MSCNN achieves **100% test accuracy** on their dataset, outperforming alternatives (a WPD-MSCNN using wavelet packets, and a standard CNN). In particular, they showed the MSCNN with SSA-VMD had a test accuracy of \~100%, vs \~93.8% for the WPD-MSCNN. The authors note that extracting multi-scale spectral features via SSA-VMD effectively separates cavitation noise from background.

**Mathematical details:** Let the raw signal be \$x(t)\$. SSA produces components \$x\_{\text{trend}}(t)\$ and \$x\_{\text{fluc}}(t)\$. Then VMD solves:

```math
\{u_k(t),\omega_k\} = \arg\min \sum_{k=1}^K \Big\| \partial_t [u_k(t)e^{-j\omega_k t}] \Big\|_2^2 
\quad \text{subject to}\quad \sum_{k=1}^K u_k(t) = x_{\text{fluc}}(t). 
```

The modes \$u\_k\$ are then normalized and fed to the CNN. A multiscale CNN applies convolutions

```math
f_{i,j}^{(l)} = \sigma\Big( \sum_{m=1}^{M} w_{i,m}^{(l)}\,x_{j+m-1}^{(l-1)} + b_i^{(l)}\Big)
```

on each channel, with ReLU activation \$\sigma(\cdot)\$. The outputs are pooled (max or average) via

```math
p_{i,j}^{(l)} = \max_{0\le m < s} f_{i,sj+m}^{(l-1)}, \quad
a_{i,j}^{(l)} = \frac{1}{s}\sum_{m=0}^{s-1} f_{i,sj+m}^{(l-1)}.
```

Finally, a fully-connected layer leads to a softmax binary prediction.

This SSA-VMD-MSCNN method excels in precision (virtually 100% on tested cases) at the cost of relatively heavy pre-processing. It demonstrates the benefit of physics-inspired feature extraction before classification.

### B.2 State Classification using SSA-VMD and CNNs

Some works extend binary detection to multi-state classification (e.g. non-cavitation, incipient, developed, severe cavitation). The general approach still applies SSA and/or VMD to decompose the signal, then a CNN for classification. For example, if the labels are \$y\in{0,1,2,3}\$ for four states, the CNN’s final layer uses a 4-way softmax.

While the SSA-VMD-MSCNN paper \[Li *et al.*] focused on binary labels, its architecture can naturally generalize: the softmax output would have as many neurons as classes. The features (VMD modes, energies) capture changes as cavitation develops. In practice, labels are often defined by cavitation number or visual inspection into discrete states.

### B.3 Cavitation Detection and Intensity Recognition using 1-D DHRN Multitask Networks

Sha *et al.* (2022) addressed cavitation detection and *intensity recognition* simultaneously from **valve acoustic signals**. They employed a **1-D Double Hierarchical Residual Network (DHRN)** in a multi-task learning setup:

* **Data Augmentation (Swin-FFT):** To compensate limited data, they introduced a sliding-window FFT method (“Swin-FFT”). Each long acoustic sample is split into overlapping segments (window size \$W\$). Each segment is transformed via FFT to the frequency domain. This increases the dataset and filters out low-frequency drift. Formally, for a time series \$x\[n]\$, windows \$x\[n\:n+W-1]\$ are taken and transformed to magnitude spectra \$X\[f]\$.

* **1-D DHRN Architecture:** The network is 1-D CNN-based with *double hierarchical residual blocks (DHRB)*. A 1-D DHRB (Fig.7 of Sha *et al.*) contains two convolutional layers with large kernels (e.g. size 32 and 16) and two parallel skip connections. One skip is the identity, the other passes through an extra 1×1 Conv. The block output is a concatenation of these paths, providing multi-scale feature fusion. An example DHRB transforms an input vector \$x\$ by:

  * Path1: identity \$x\$.
  * Path2: conv-\$k\_1\$ + BN + ReLU, conv-\$k\_2\$ + BN + ReLU.
  * Output: \$y = \[x, F(x)]\$ concatenated.

From Sha *et al.*'s details, if \$F(x)\$ is the residual branch output, then

```math
y = \begin{bmatrix} x \\ F(x) + x \end{bmatrix} \in \mathbb{R}^{2d},
```

where \$d\$ is the channel dimension. The 1-D DHRN uses successive DHRBs (18 layers total) with pooling interleaved (see Table 2 in \[Sha *et al.*). The network splits into two heads for multitask output: one sigmoid or softmax for binary detection, another (stretched sigmoid or similar) for intensity regression (label 0–9). They found DHRN outperformed plain CNN or GRU on this acoustic task.

* **Results:** Using their augmented data, the 1-D DHRN achieved high accuracy. For detection (binary), they report up to **100%** accuracy on the cleanest dataset, and \~97% on others. For intensity (multi-class 0–9), accuracies ranged \~94–100% on each of three datasets. This shows that large 1-D conv kernels and residual connections capture relevant patterns in valve signals. The Swin-FFT augmentation was crucial: it effectively combats noise and limited samples by providing many FFT segments.

### B.4 Data Augmentation (Swin-FFT) and 1-D DHRN Structure

We detail the Swin-FFT method and 1-D DHRN:

* **Swin-FFT (Sliding Window + FFT):** Each recorded acoustic waveform \$x(t)\$ is long (hundreds of milliseconds). To augment, a window of length \$W\$ slides with some overlap. Each windowed segment \$x\_w(t)\$ (length \$W\$ samples) is FFT-transformed:

  ```math
  X_w(f) = \sum_{n=0}^{W-1} x_w[n]\cdot e^{-j 2\pi fn/W}.
  ```

  The magnitude \$|X\_w(f)|\$ is used as an input feature vector. By choosing \$W\$ smaller than the full signal length, this creates many overlapping FFT samples, increasing data and reducing noise variance. The only hyperparameter is the window size \$W\$. In practice, Sha *et al.* tuned \$W\$ (around tens of ms) to balance segment independence and information content.

* **1-D DHRN architecture:** The overall 1-D DHRN (Figure 1 of Sha *et al.*) begins with one convolutional layer (kernel length 32, 64 filters), followed by multiple DHRBs with pooling. The detailed layers (from \[Sha *et al.*) were:

  1. Conv-1: \$64\$ filters, kernel \$32\$, stride 1.
  2. DHRB-1: Residual block with kernels \[32,16], output 64 channels (stride 1).
  3. Pool (stride 2).
  4. DHRB-2: \[32,16] block, output 128 channels (stride 1).
  5. Pool.
  6. DHRB-3: \[32,16], output 256 (stride 1).
  7. Pool.
  8. DHRB-4: \[32,16], output 512 (stride 1).
  9. Pool.
  10. Flatten and dense layers.

Table 2 in  lists these sizes. Convolution (4) and pooling (5),(6) from \[Sha *et al.* are:

```math
f_{i,j}^{(l)} = \sigma\Big(\sum_{m=0}^{M-1} w_{i,m}^{(l)}\,x_{j+m}^{(l-1)} + b_i^{(l)}\Big), \quad
p_{i,j}^{(l)} = \max_{0\le m<s} f_{i,sj+m}^{(l-1)}, \quad
a_{i,j}^{(l)} = \frac{1}{s}\sum_{m=0}^{s-1} f_{i,sj+m}^{(l-1)},
```

where \$s\$ is pooling size. For example, with \$s=2\$, each pooling halves temporal dimension.

In summary, the 1-D DHRN with Swin-FFT achieved both high accuracy and robust recognition of cavitation state and intensity. Its large kernels and double shortcuts capture long-range features in signals.

### B.5 Knowledge Distillation CNN (KD-CNN)

Liu *et al.* (2025) proposed a knowledge distillation approach (called KD-CNN) to balance speed and accuracy. They addressed the problem that complex CNNs have high accuracy but slow inference, whereas simple CNNs are fast but less accurate.

* **Teacher model:** A relatively deep CNN (3 convolutional layers + pooling + dense) is trained on acoustic emission data (from *acoustic emission* sensors on turbine blades, not hydroacoustic, but principle is similar). Each conv layer uses 64 filters of kernel size 10 and stride 1, followed by 2×2 max-pooling (see Table 2 in \[37]). For example:

  * Input: 2000-sample AE signal.
  * Conv1: 64 filters (10×1), output feature 1991×64, then pool → 995×64.
  * Conv2: 64 filters, pool → 493×64.
  * Conv3: 64 filters, pool → 242×64.
  * Fully-connected, softmax (number of classes).
    This teacher CNN achieved very high accuracy on training data.

* **Student model:** A simpler CNN with just **one** convolutional layer (plus pooling and dense) is defined. It is much smaller, enabling faster inference.

* **Knowledge Distillation:** After training the teacher on labeled data, the *soft outputs* of the teacher are used as targets for the student. Specifically, the original labels are replaced with the teacher’s output probabilities. The student CNN is then trained to **minimize cross-entropy** against these soft labels. Formally, if the teacher’s output for sample \$i\$ is a probability vector \$t^{(i)} = (t^{(i)}\_1,\dots,t^{(i)}\_C)\$ over classes, and the student’s output is \$s^{(i)}\$, the student minimizes:

  ```math
  \mathcal{L}_{KD} = -\sum_{i=1}^N \sum_{k=1}^C t_k^{(i)} \log s_k^{(i)},
  ```

  which encourages the student to mimic the teacher. No explicit temperature is mentioned, but the idea is the same as Hinton *et al.*’s distillation.

* **Performance:** The resulting KD-CNN (student) achieves near the teacher’s accuracy while being much simpler. Liu *et al.* report that the KD-CNN can recognize cavitation state in under 2 seconds, with each condition’s accuracy above 97%. This demonstrates that distillation yields a model with “the recognition speed of the student and the accuracy of the teacher”.

This method is useful when real-time decision (fast inference) is needed without sacrificing accuracy.

### B.6 AC-GAN-Based Architectures (Robust Classification & Data Augmentation)

Generative adversarial networks (GANs) have been applied to cavitation detection for two purposes: **data augmentation** and **robust classification**. In particular, the Auxiliary Classifier GAN (AC-GAN) architecture has been used:

* **Auxiliary Classifier GAN (AC-GAN):** In a standard GAN, a generator \$G\$ learns to map noise \$z\sim p\_z\$ to fake samples \$\tilde{x}=G(z)\$, while a discriminator \$D\$ learns to distinguish real vs. fake. In an *AC-GAN*, \$G\$ additionally takes a class label \$c\$ as input, so \$G(z,c)\$ generates a sample of class \$c\$. The discriminator \$D\$ has two outputs: (1) the probability that the input is real or fake (\$S\in{\text{real},\text{fake}}\$) and (2) the predicted class label \$C\$. Training objectives involve two losses: a source loss \$L\_S\$ for real/fake and a class loss \$L\_C\$ for correct label. From Look *et al.*, these are:

  ```math
  L_S = \mathbb{E}[\log P(S=\text{real}\mid X_{\text{real}})] + \mathbb{E}[\log P(S=\text{fake}\mid X_{\text{fake}})],
  $$$
  $$  
  L_C = \mathbb{E}[\log P(C=c\mid X_{\text{real}})] + \mathbb{E}[\log P(C=c\mid X_{\text{fake}})],
  ```

  where \$X\_{\text{real}}\$ are real cavitation/noise samples and \$X\_{\text{fake}}=G(z,c)\$.  After training, the discriminator serves both as a classifier and an anomaly detector. In practice, the discriminator’s predicted class \$C\$ is used for cavitation detection.

* **Training Procedure:** The generator \$G\$ is trained to maximize both \$L\_C\$ (so it generates samples correctly classified) and to confuse \$D\$ on real/fake. The discriminator maximizes \$L\_S+L\_C\$. Look *et al.* (ICPRAM 2018) observed that training an AC-GAN on acoustic spectrograms of cavitation led to a robust classifier with better generalization. They used a KL-divergence penalty \$I(X|Y)\$ between spectrograms to stabilize training.

* **Robust Classification:** The key advantage of AC-GAN is that the trained discriminator acts as a powerful multi-task classifier (real/fake and class). After convergence, one can drop the generator and use \$D\$ alone for cavitation detection. Look *et al.* found that incorporating AC-GAN improved the classifier’s robustness to sensor placement and turbine differences.

* **Data Augmentation:** The generator \$G\$ produces new synthetic spectrograms for each class. These can augment the training set of a CNN. For example, training a CNN on real+GAN-generated data increased accuracy.

Overall, AC-GAN yields an architecture where:

```math
\text{Discriminator } D:\; x \mapsto (P(S=\text{real}),\,P(C=c)), 
\quad \text{Generator } G(z,c): z\sim \mathcal{N}(0,I)\mapsto \tilde{x}.
```

After training, \$D\$ is a “robust classifier”. AC-GAN training often boosts binary cavitation detection accuracy (e.g. from \~94% to \~98%) while also supplying synthetic samples.

### B.7 CNN Training with Acoustic Spectrograms (Enhanced by GAN)

Another line of work trains CNNs directly on time-frequency representations of cavitation acoustic signals. The process is:

1. **Spectrogram computation:** High-frequency acoustic signals (e.g. 100–1000 kHz band) are segmented and STFT is applied (often with Hanning windows). Then dynamic-range compression is applied: $f(x) = \log(1+Cx)$ to emphasize low-energy events. Optionally, a filter bank or Mel-scaling reduces dimension.

2. **CNN architecture:** A common choice is a VGG-like 2D CNN, with alternating convolution and max-pooling layers. Look *et al.* (Cav2018) used 3×3 or 3×3 kernels, doubling filters across layers (8→16→32). Dropout (e.g. 0.2) and early stopping were employed to prevent overfitting. The CNN was trained to classify 50/50 cavitation vs. no-cavitation spectrograms.

3. **Performance:** With these settings, the CNN achieved about **94.2%** detection accuracy on held-out turbines. This serves as a baseline for cavitation detection via spectrogram + CNN.

4. **Enhancement with AC-GAN:** As described above, injecting GAN-generated spectrograms into training improved accuracy. In Look *et al.* (Cav2018), after training an AC-GAN on spectrograms, they used the discriminator as a final predictor. This raised accuracy to **98.2%**. They interpreted the generator \$G\$ as a data augmenter: it produces diverse spectrogram patterns for both classes.

Mathematically, if \$X\$ is the spectrogram image (a matrix of STFT magnitudes), the CNN computes:

```math
h = \text{CNN}(X;\theta), \quad \hat{y} = \text{softmax}(W h + b),
```

with cross-entropy loss

```math
\mathcal{L}_{CE} = -\sum_{i=1}^N \sum_{k} y_k^{(i)} \log \hat{y}_k^{(i)}.
```

After including GAN samples, the effective training set is enlarged. The AC-GAN yields a modified loss (as above) but ultimately, \$D\$’s classification output is used.

The key takeaway: 2-D CNNs on spectrograms can effectively detect cavitation, and GANs significantly increase their robustness and accuracy.

### B.8 Combining Multi-modal Signals (Including Electromagnetic)

While most studies focus on acoustic or vibration signals, some approaches combine multiple sensors. For example, an AC-GAN classifier by Look *et al.* was designed to work “independent of sensor position and turbine type”, implying use of multiple hydrophones. Other research (outside the neural-net paradigm) has explored electromagnetic or pressure sensors to detect cavitation indirectly (by measuring effects on the electromagnetic field or structural vibrations). These multi-modal approaches are still nascent. In neural-network terms, multi-modal fusion could mean concatenating features from different sensors or using separate CNN branches. Data fusion could improve robustness to noise, but specifics in the literature are limited.

One strategy is feature-level fusion: e.g., perform VMD on acoustic and on vibration signals, compute envelope features for both, and feed into a joint CNN. Another is decision-level fusion: run separate classifiers on each modality and fuse their outputs. The cited works emphasize that acoustic data alone can achieve high accuracy, but combining modalities remains an open area. For completeness, modern pipelines could include electromagnetic or structural health sensors as additional channels, but standard datasets and studies are not yet prevalent in the literature we surveyed.

## C) Comparative Analysis of Methods

We summarize and compare the above methods along key criteria: accuracy (or classification performance), inference speed (model complexity), and robustness (resilience to noise, sensor variability).

* **Accuracy:** Empirical results show that dedicated CNN approaches achieve very high accuracy on benchmark datasets. For instance, Li *et al.*’s SSA-VMD-MSCNN attained \~**100%** binary detection accuracy on their test set. Sha *et al.*’s 1-D DHRN achieved detection accuracy of **97–100%** and intensity recognition up to **100%** on three test sets. Look *et al.*’s baseline CNN on spectrograms got **94.2%**, which was improved to **98.2%** with AC-GAN augmentation. The KD-CNN student consistently achieved >97% for each cavitation state. These figures indicate that CNN-based models can nearly saturate accuracy under controlled conditions. However, slight differences arise: the SSA-VMD approach may excel when the signal model matches assumptions, while AC-GAN/QGAN-enhanced CNNs may generalize better in cross-machine tests.

* **Inference speed:** Model complexity varies. SSA-VMD-MSCNN has heavy preprocessing (SSA and VMD, which are iterative algorithms), so feature extraction is relatively slow. The CNN itself is multi-channel with multiple conv layers, moderate speed. The 1-D DHRN has 18 layers but 1-D convolutions (which are faster than 2-D), still moderate. The baseline CNN in Look’s pipeline (few layers, small filters) is fast. The KD-CNN’s student has only 1 conv layer and is very fast (Liu *et al.* achieve recognition within \~2 seconds for a dataset). AC-GAN training is slow, but inference uses only the discriminator (a CNN with a handful of conv layers, similar to the baseline CNN), so speed is comparable to standard CNN. In summary, the ranking (slowest→fastest) is: **SSA-VMD (due to VMD)**, then **deep CNNs (e.g. DHRN)**, then **standard CNN**, and **distilled student CNN** fastest.

* **Robustness:** Robustness refers to performance stability under noise, unseen conditions, and sensor variability. In this regard, methods incorporating data augmentation or domain adaptation fare better. The AC-GAN-based method explicitly aimed for generalization across turbines. Gaisser *et al.*’s domain-alignment CNN explicitly handles domain shifts via adversarial training, producing a classifier that works on various machines. The SSA-VMD approach is resilient to some noise (by decomposing and focusing on cavitation bands), but its reliance on parameter tuning (e.g. number of VMD modes) could be brittle. The 1-D DHRN with Swin-FFT was shown to filter noise effectively.  KD-CNN’s robustness mainly comes from the teacher training on broad conditions; the student then inherits this robustness. Quantitatively, Look *et al.* achieved consistent accuracy on separate turbines for detection, and KD-CNN achieved >97% across multiple conditions. A key point: methods using adversarial or GAN techniques explicitly target robustness (domain shift, sensor variation), whereas standard CNNs may degrade off-distribution.

Table 1 (conceptual) compares the methods:

* **SSA-VMD-MSCNN:** Accuracy ≈100%, preprocessing-heavy (slow), medium robustness (depends on signal stationarity).
* **SSA-VMD + CNN (multi-state):** Similar to above, accuracy and complexity scale with classes; effective for multiple cavitation levels.
* **1-D DHRN (multitask):** Accuracy \~97–100%, 1-D CNN is fast, high robustness via augmentation, handles intensity regression.
* **Swin-FFT:** Augments data well, no direct accuracy metric (enables others).
* **KD-CNN:** Accuracy >97%, extremely fast inference, good robustness inherited from teacher.
* **AC-GAN:** Baseline CNN \~94%, augmented \~98%, moderate speed, very robust to domain differences.
* **CNN on spectrogram:** \~94% base, simple and fast, moderate robustness (improved with GAN/data).
* **Domain-adaptive CNN:** (Gaisser *et al.*) Accuracy not given here, but specifically designed for cross-machine generality.

Overall, methods combining signal processing (SSA-VMD) with CNNs yield high precision in controlled tests, while GAN-based and distillation approaches excel at trading off speed and robustness. For implementation, one must consider the target environment: for laboratory turbine monitoring, high-complexity models (SSA-VMD) might be used offline. For real-time online detection, a distilled CNN or GAN-enhanced CNN is attractive due to fast inference.

## D) Synthetic Data Generation and Augmentation

Given the difficulty of collecting large cavitation datasets, synthetic data generation has become important. Two main strategies are: (1) *Model-based simulation* (physics-driven), and (2) *Data-driven generative models* (GANs, VAEs). Here we focus on GAN-based augmentation since it integrates with neural classifiers.

### D.1 Generative Adversarial Networks (GANs)

**Standard GAN:** A basic GAN has a generator \$G\$ and discriminator \$D\$.  \$G(z;\theta\_G)\$ maps random noise \$z\sim p\_z\$ to synthetic sample \$\tilde{x}\$. The discriminator \$D(x;\theta\_D)\$ outputs \$D(x)=P(\text{real}|x)\$. They play a minimax game:

```math
\min_G \max_D \; \mathbb{E}_{x\sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z\sim p_z}[\log(1 - D(G(z)))].
```

Training alternates updating \$D\$ (to maximize correct real/fake classification) and \$G\$ (to fool \$D\$). After training, \$G\$ can synthesize new data.

**AC-GAN (Auxiliary Classifier GAN):** As described, this conditions on class labels. The discriminator outputs both source and class. The loss functions are the \$L\_S,L\_C\$ given before. After training, sampling is done as \$G(z,c)\$ where \$z\$ is random and \$c\$ a desired cavitation class (e.g. “cavitation present” or “absent”).

### D.2 Other Generative Models

* **Conditional GAN (cGAN):** Similar to AC-GAN but without an auxiliary loss; the label is input to \$G\$ and \$D\$. It can generate samples of a given class.

* **Wasserstein GAN (WGAN):** Replaces the Jensen-Shannon loss with a Wasserstein distance to improve training stability. WGAN can also be conditioned.

* **Variational Autoencoder (VAE):** Learns a latent representation \$z\$ from data; by sampling \$z\$ one can generate new data. VAEs tend to produce blurrier outputs than GANs, so are less common for spectrograms.

* **Data Augmentation Techniques:** Besides GANs, classical augmentation (addition of noise, random shifts, time-stretching, pitch scaling) can be applied to acoustic signals. For instance, random noise or small time delays create new training examples. The Swin-FFT method itself is a form of augmentation.

### D.3 Augmentation Pipelines

A typical augmentation pipeline for acoustic signals might include:

1. **Time-Domain Augmentations:** Random cropping, additive Gaussian noise, slight time stretching (to simulate speed variations), or filtering.
2. **Frequency-Domain Augmentations:** On spectrograms, apply random frequency/time masking (SpecAugment style), or overlay random noise segments.
3. **GAN Synthesis:** Train an AC-GAN on existing cavitation spectrograms (including both cavitation and non-cavitation classes). Once trained, generate additional labeled spectrograms for the minority class to balance the dataset.
4. **Mixup:** Combine two spectrograms linearly (with interpolation) to create new intermediate examples.

Each generated spectrogram can be visualized (as an example). For instance, Figure 1 (not shown) could depict a spectrogram of a cavitating vs. a non-cavitating signal. (In a text-only format we cite relevant equations instead.)

### D.4 Example Spectrogram (Illustrative)

*No actual figure is shown here, but for context:* An acoustic spectrogram of cavitation noise typically shows broadband energy around certain collapse frequencies. GAN-generated spectrograms aim to mimic these patterns.

In summary, augmentation by GANs significantly enriches training sets. The generated data obey the learned time-frequency distribution of cavitation signals. Mathematically, if \$D\$ is well-trained, sampling \$x' = G(z,c)\$ for many \$z\$ yields a dataset \${x'*i}\$ that approximates the real distribution \$p*{\text{data}}(x|c)\$. These samples are then used to train or fine-tune the CNN discriminator/classifier.

## E) Conclusions

Neural network methods for cavitation detection in hydraulic turbines have matured considerably. Key findings:

* **Signal Decomposition + CNN:** Combining SSA or VMD with CNNs (MSCNN, multi-index fusion) yields very high accuracy on clean data. These methods explicitly incorporate domain knowledge (vapor bubble frequencies) and excel in controlled environments.
* **Multitask and Hierarchical CNNs:** 1-D DHRN architectures successfully perform simultaneous cavitation detection and intensity estimation. Their large kernels capture relevant patterns in acoustic signals.
* **Knowledge Distillation:** KD allows a simple CNN to run in real time with near-elite accuracy. This is promising for online monitoring where computational resources are limited.
* **GANs and Data Augmentation:** AC-GANs serve a dual role: generating synthetic cavitation spectra and acting as robust classifiers. Use of GAN-augmented training data pushes CNN accuracy higher (e.g. \~98%) and reduces dependence on large labeled datasets.
* **CNN on Spectrograms:** A straightforward CNN on spectrogram inputs can achieve \~94% detection accuracy, showing that deep learning can supplant hand-crafted feature methods.
* **Robustness & Generalization:** Methods explicitly designed for domain shift (e.g., adversarial domain alignment) produce models that generalize across turbine models. This is critical for real-world deployment across different plants.

In practice, a *fusion* approach may be optimal: use SSA-VMD or filterbank features to enhance signal, then feed into a CNN. Employ GAN-based augmentation to enlarge training sets. Use knowledge distillation to compress the model. And include any available sensor modalities (acoustic + vibration) for ensemble detection.

Engineering implementation should balance accuracy and speed. A distilled 1-D CNN (like the KD-CNN student) can run in real time, detecting cavitation state every few seconds. For more nuanced monitoring (intensity level), a larger model (like DHRN) can be used offline or with dedicated hardware.

Finally, maintaining a diverse training dataset (from different operating conditions and turbines) is essential. Continual learning and online adaptation (possibly via incremental GAN retraining) can help cope with changing environments. The survey shows that, under ideal conditions, accuracy can approach 100%, but robustness to noise and domain shift remains the primary challenge. Future work will likely explore self-supervised or domain-general techniques to handle these challenges.

## F) References

1. F. Li, X. Song, X. Li, X. Feng et al., “A novel cavitation diagnosis method for hydraulic turbines based on SSA, VMD and multiscale CNN,” *Ocean Engineering*, vol. 312, p. 119055, 2024.
2. Y. Sha, T. Wang, Z. Li, S. Sun and Y. Jiang, “A multi-task learning for cavitation detection and cavitation intensity recognition of valve acoustic signals,” *Eng. Appl. Artif. Intell.*, vol. 113, 2022, Art. 104904.
3. Z. Liu, Z. Zhou, S. Zou, Z. Liu, and S. Qiao, “Cavitation state identification method of hydraulic turbine based on knowledge distillation and convolutional neural network,” *Power Gener. Technol.*, vol. 46, no. 1, pp. 161–170, 2025.
4. A. Look, O. Kirschner, and S. Riedelbauch, “Building robust classifiers with generative adversarial networks for detecting cavitation in hydraulic turbines,” in *Proc. Int. Conf. Pattern Recogn. Appl. Meth. (ICPRAM)*, 2018, pp. 456–462.
5. A. Look, O. Kirschner, S. Riedelbauch, and J. Necker, “Detection and level estimation of cavitation in hydraulic turbines with convolutional neural networks,” in *Proc. 12th Int. Symp. Cavitation*, 2018, pp. 543–550.
6. L. Harsch, O. Kirschner, and S. Riedelbauch, “Cavitation detection in hydraulic machinery under strong domain shifts using neural networks,” *Phys. Fluids*, vol. 35, no. 2, 2023, Art. 027128.
7. Y. Wang, F. Li, M. Lv, T. Wang, and X. Wang, “A multi-index fusion adaptive cavitation feature extraction for hydraulic turbine cavitation detection,” *Entropy*, vol. 27, no. 4, Art. 443, 2025.
8. J. H. Gaisser, O. Kirschner, and S. Riedelbauch, “Domain-adversarial cavitation detection in hydraulic turbines,” *Phys. Fluids*, (accepted for publication, 2023).
9. D. Petrescu et al., “Adaptive multi-index feature fusion for reliable cavitation monitoring,” *MDPI J.* (in press), 2024.
10. E. Dragomiretskiy and D. Zosso, “Variational mode decomposition,” *IEEE Trans. Signal Process.*, vol. 60, pp. 654–664, 2012.

* (The above references correspond to the cited sources in this text.)
