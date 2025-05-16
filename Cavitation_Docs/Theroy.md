# Technical Report on Cavitation Detection in Hydraulic Turbines Using Neural Networks

## Introduction

This report brings together results from multiple peer-reviewed studies to offer a unified overview of modern diagnostic techniques. Key mathematical formulations—ranging from Sparrow Search Algorithm–optimized Variational Mode Decomposition (SSA-VMD) through multiscale and hierarchical convolutional architectures to knowledge-distilled and adversarially trained networks—are presented here in concise, summary form. For clarity and brevity, each equation is stated in its essential form; full derivations and extended discussions may be found in the original references. By consolidating these methods, this compendium aims to equip researchers and practitioners with a clear roadmap for implementing both real-time binary cavitation alarms and multiclass severity classification in hydroelectric machinery.

## A. Detection and Classification Methods

### SSA-VMD-MSCNN Method

The SSA-VMD-MSCNN approach integrates *Variational Mode Decomposition* (VMD) optimized by the *Sparrow Search Algorithm* (SSA) with a multiscale convolutional neural network (MSCNN) for binary cavitation detection from hydroacoustic signals [scrib.com](https://www.scribd.com/document/766299017/1-s2-0-S002980182402393X-main#:~:text=CNN%20to%20adaptively%20decompose%20nonstationary,diagnostic%20performance). VMD is formulated as a constrained variational problem: it seeks modes $u_k(t)$ and center frequencies $\omega_k$ to minimize the sum of bandwidths subject to reconstructing the signal $x(t)$. Formally, following Dragomiretskiy and Zosso (2013), the VMD optimization can be stated as

```math
\min_{\{u_k\},\{\omega_k\}} \sum_{k=1}^K \left\| \partial_t\Bigl[(\delta+\tfrac{j}{\pi t}) * u_k(t)\Bigr] e^{-j \omega_k t}\right\|_2^2 
\quad\text{s.t.}\quad \sum_{k=1}^K u_k(t) = x(t).
```

An augmented Lagrangian with multiplier $\lambda(t)$ is introduced, leading to an iterative update rule for each mode in the Fourier domain. In practice, given an estimated Lagrange multiplier $\hat{\lambda}(\omega)$ and other modes ${\hat{u}_i}$, each mode $\hat{u}_k(\omega)$ is updated by:

```math
\hat{u}_k^{(n+1)}(\omega) = \frac{\hat{x}(\omega) - \sum_{i \neq k}\hat{u}_i^{(n)}(\omega) + \tfrac{1}{2}\hat{\lambda}^{(n)}(\omega)}{1 + 2\alpha_k(\omega - \omega_k^{(n)})^2},
```

where $\alpha_k$ is the penalty factor for mode $k$ and $\hat{x}(\omega)$ is the Fourier transform of the input. The process cycles through modes and Lagrange multipliers until convergence. The Sparrow Search Algorithm (SSA) is used to optimize the VMD hyperparameters $K$ (number of modes) and $\alpha_k$ by minimizing a fitness function (e.g. reconstruction error). This results in adaptively decomposed hydroacoustic signals into intrinsic mode functions (IMFs) capturing different characteristic scales.

Once decomposition is performed, each IMF (or zero-padded channel) is fed into a multiscale CNN. The MSCNN consists of parallel branches of 1-D convolutional layers with ReLU activations, followed by pooling. A single convolutional layer computes feature maps via discrete convolution: for input feature map $f^{(l-1)}$ and convolution kernel $w$, the output feature $f^{(l)}$ is

```math
(f^{(l)} * w)[n] = \sum_{m} f^{(l-1)}[n-m]\,w[m] + b,
```

followed by ReLU activation $h(z)=\max(0,z)$. The pooling layer performs downsampling (e.g. max pooling) by taking the maximum over a sliding window. Mathematically, if $v$ is a 1-D feature vector and the pooling stride is \$s\$, then

```math
\text{maxpool}(v)_i = \max_{1 \le j \le p} v[(i-1)s + j],
```

where $p$ is the pooling width. After several convolution + pooling layers (depth $L$), channel outputs are concatenated into a feature vector. A fully-connected layer then maps to logits, and a softmax outputs class probabilities:

```math
p(y=k|x) = \frac{e^{z_k}}{\sum_{i} e^{z_i}}.
```

Cross-entropy loss is used for binary classification.

**Algorithm (pseudocode):** The training process for SSA-VMD-MSCNN can be outlined as follows:

```python
# Pseudocode: SSA-VMD-MSCNN training
class SSA_VMD_MSCNN(nn.Module):
    def __init__(self, K, alpha_init, cnn_params):
        super().__init__()
        self.K = K
        self.alpha = alpha_init
        self.cnn = MultiscaleCNN(**cnn_params)
    def forward(self, x):
        # SSA optimizes VMD parameters (K and alpha)
        optimal_params = SSA_optimize(self.alpha, data=x)
        # VMD decomposition with optimal parameters
        imfs = VMD_decompose(x, K=self.K, alpha=optimal_params)
        # Stack IMFs as separate channels (or apply padding)
        multi_channel = torch.stack(imfs, dim=1)  # shape: (batch, K, length)
        # CNN forward on multiscale channels
        features = self.cnn(multi_channel)
        logits = self.cnn.fc(features)
        return logits

# Training loop
model = SSA_VMD_MSCNN(K=10, alpha_init=..., cnn_params=...)
optimizer = torch.optim.Adam(model.parameters())
for epoch in range(num_epochs):
    for batch_x, batch_y in train_loader:
        logits = model(batch_x)
        loss = CrossEntropyLoss(logits, batch_y)
        loss.backward(); optimizer.step(); optimizer.zero_grad()
```

In this design, SSA searches for optimal VMD parameters (line by line in Algorithm 1 of Li et al. ) and VMD then adaptively decomposes the signal. The MSCNN (multiscale CNN) learns features concurrently from all mode channels. This method was shown to outperform traditional CNNs and wavelet-packet CNNs in diagnosing turbine cavitation by effectively capturing nonstationary, multiscale hydroacoustic features.

### Hydroacoustic State Classification with SSA-VMD + MSCNN

Beyond binary cavitation detection, SSA-VMD+MSCNN can be extended to multiclass state classification (e.g. non-cavitation, incipient, severe) by adjusting the final softmax for multiple outputs. The procedure remains similar: hydrophone signals are decomposed via SSA-optimized VMD into IMFs at different frequency bands, then these components feed into a multiscale CNN. Each channel learns features specific to its band, and the fused representations distinguish among multiple cavitation states. This approach removes the need for hand-crafted features or fixed wavelet filters. In practice, one simply replaces the binary cross-entropy with a multiclass cross-entropy loss. The **key advantage** is adaptivity: the optimized VMD layer ensures frequency bands align with the signal’s characteristics, making the CNN robust to variations in flow and noise. For example, Li et al. report that SSA-VMD-MSCNN accurately detected airfoil cavitation (a common Francis turbine fault) from raw hydroacoustic data under varying conditions. Mathematically, the multiclass output is

```math
p(y = c | x) = \frac{\exp(z_c)}{\sum_{i} \exp(z_i)},\quad c=1,\dots,C,
```

with loss $-\sum_c y_c \log p(y=c|x)$.

### DHRN-Based Cavitation Detection (1-D Double Hierarchical Residual Networks)

A 1-D Double Hierarchical Residual Network (DHRN) is another powerful model for cavitation detection and classification. It stacks *Double Hierarchical Residual Blocks* (DHRBs) in a CNN architecture. Each 1-D DHRB contains two convolutional layers with **different kernel sizes** and two shortcut connections: one identity, and one involving a $1\times 1$ conv to match dimensions. If we denote the input to a DHRB by $x$, the block computes two parallel paths: a large-kernel conv ($k_1$) and a smaller conv ($k_2$), plus two residual shortcuts. Formally, letting $F(x)$ be the output of the first large conv+ReLU+BN stack and $G(x)$ the second small conv+ReLU+BN stack, a DHRB outputs

```math
y = [\,x + F(x)\,]\;||\;[\,H(x) + G(F(x))\,],
```

where $H(x)$ is the shortcut conv (if needed) and $||$ denotes concatenation. Concretely, Sha et al. used kernel sizes of 32 and 16 (with size-32 first, then size-16) and ReLU activations. The two shortcuts ensure gradient flow and capture both identity and transformed features. Many DHRBs of varying filter counts are stacked, preceded by an initial Conv+ReLU and max-pool, and followed by global average pooling and fully-connected output layers. The authors built an 18-layer network: Conv(32)->Pool, then four DHRB “layers” with increasing filter counts (64→128→256→512), each downsampling via stride=2, concluding in FC and two softmax heads for multi-task output.

The 1-D convolution operation in these networks is given by: for filter $k$ at layer $l$, the $i$-th feature is

```math
a_{k,i}^{(l)} = \sigma\Bigl(\sum_{j} w_{k}^{(l)}\cdot x^{(l-1)}_{j,i} + b_k^{(l)}\Bigr),
```

where $\sigma$ is ReLU, $x^{(l-1)}_{j,i}$ is the $i$-th element of the $j$-th input feature map, and $(w_k^{(l)},b_k^{(l)})$ are the filter weights and bias. The pooling is similarly as above.

**Training:** The 1-D DHRN is trained end-to-end with cross-entropy losses. In multi-task mode, one head predicts cavitation presence (binary), the other predicts intensity level (multiclass). Denote model output logits for binary by $z^{(2)}$ (2 classes) and for intensity by $z^{(4)}$ (4 classes), with ground truth one-hot vectors $y^{(2)}$, $y^{(4)}$. The loss is

```math
\mathcal{L} = -\sum_{i=1}^2 y^{(2)}_i \log p_i^{(2)} - \sum_{j=1}^4 y^{(4)}_j \log p_j^{(4)},
```

where $p^{(c)} = \text{softmax}(z^{(c)})$.

**Data Augmentation:** To address limited data, a **Swin-FFT** sliding-window augmentation is applied. The signal is windowed into overlapping segments, each transformed by FFT, yielding many frequency-domain samples. This increases the training set and makes the network invariant to time shifts.

**Performance:** The 1-D DHRN achieved state-of-the-art results on valve acoustic datasets: cavitation detection accuracies \~97–100% and intensity recognition 93–100%. It was also tested with real-world noise (Dataset 3) and maintained high accuracy. This demonstrates robustness: the hierarchical residual structure captures multiscale spectral features effectively.

**Pseudocode:** A PyTorch-style snippet for the DHRN block and network:

```python
class DHRB(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        # First conv (large kernel)
        self.conv1 = nn.Conv1d(in_channels, mid_channels, kernel_size=32, padding=16)
        self.bn1 = nn.BatchNorm1d(mid_channels)
        # Second conv (smaller kernel)
        self.conv2 = nn.Conv1d(mid_channels, out_channels, kernel_size=16, padding=8)
        self.bn2 = nn.BatchNorm1d(out_channels)
        # Shortcut conv for identity (to match channels if needed)
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels)
        )
    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)))
        out2 = F.relu(self.bn2(self.conv2(out1)))
        # Identity shortcut
        id1 = x
        # Convolutional shortcut
        id2 = self.shortcut(x)
        # Concatenate identity paths
        return torch.cat([id1 + out2, id2 + out2], dim=1)

# Constructing the full DHRN
class DHRN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=32, padding=16), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.layer1 = DHRB(64, 32, 64)
        self.layer2 = DHRB(128, 32, 128)  # in_channels doubled after cat
        self.layer3 = DHRB(256, 32, 256)
        self.layer4 = DHRB(512, 32, 512)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc4 = nn.Linear(512, 4)  # cavitation intensity (4 classes)
        self.fc2 = nn.Linear(512, 2)  # cavitation detection (2 classes)
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x).squeeze(-1)
        logits4 = self.fc4(x)
        logits2 = self.fc2(x)
        return logits4, logits2
```

### DHRN for Intensity Classification in Valve Signals (with Data Augmentation)

Building on the DHRN above, the valve signal application specifically targets cavitation **intensity levels** (e.g., none, small, medium, large). The 1-D DHRN is trained as a **multi-task network**: one output head performs binary detection, and a second head assigns one of four intensity levels. The Swin-FFT augmentation (as pseudocoded above) is crucial due to scarce measured data. Each raw time signal is segmented by a sliding window and transformed via FFT to produce the frequency-domain samples used for training.

The hierarchical residual blocks capture frequency-domain features: the larger kernel conv (size 32) extracts broad spectral patterns, while the smaller kernel (size 16) refines local frequency features. The dual shortcuts ensure information flow and flexibility. Table 2 in Sha et al. details the network: initial conv (kernel 32, 64 filters), four DHRB stages (\[32,16] kernels) with 64→128→256→512 filters, then global pooling and two FC layers. Such a network depth (≈18 layers) allows it to distinguish subtle differences in acoustic spectra.

Overall, the DHRN with Swin-FFT achieves highly accurate intensity classification (up to 100% on some datasets) and improves over single-task or shallower CNNs. The augmentation improves generalization, effectively increasing dataset diversity and reducing overfitting.

### KD-CNN (Knowledge Distillation CNN)

A **Knowledge Distillation** CNN (KD-CNN) uses a large “teacher” network to train a lightweight “student” network. In the context of turbine cavitation, Zhong Liu et al. proposed a three-layer CNN as the teacher and a single-layer CNN as the student for multiclass cavitation-state classification. The process is:

1. **Train Teacher:** The teacher CNN (deeper, e.g. 3 convolutional layers with pooling) is trained on the labeled dataset \$(X,Y)\$ to achieve high accuracy. Its architecture (illustrated in Fig. 1) might be: Conv(10×1,64 filters)→Pool→Conv(10×1,64)→Pool→Conv(10×1,64)→Pool→Softmax(4 classes).
2. **Generate Soft Labels:** Use the trained teacher to predict probabilities \$P^T = \text{softmax}(T(x))\$ for each training input \$x\$. These *soft targets* (rich in information about class similarities) replace the hard labels.
3. **Train Student:** Train the student CNN (shallower) on \$(X,P^T)\$, minimizing cross-entropy \$-\sum\_i P^T\_i\log P^S\_i\$, where \$P^S\$ are student outputs. The student network has only one conv layer (e.g. Conv(10×1,64)→Pool→Softmax(4)).
4. **Inference:** The trained student (KD-CNN) is used for real-time cavitation state recognition.

By distillation, the student learns to mimic the teacher’s function with far fewer layers. In their experiments, the KD-CNN *student* completed cavitation-state identification in \~2 seconds per instance, achieving >97% accuracy (comparable to the teacher).

**Mathematically**, knowledge distillation loss can be written as:

$$
\mathcal{L} = -\sum_{i=1}^C P^T_i \log P^S_i,
$$

where \$P^T\$ are teacher probabilities and \$P^S\$ are student probabilities over \$C\$ classes. If a temperature \$T\$ is used in softmax, \$P^T\_i = \frac{\exp(z^T\_i/T)}{\sum\_j \exp(z^T\_j/T)}\$, but in Liu et al. the standard softmax (\$T=1\$) was applied.

**Pseudocode:** Training the KD-CNN:

```python
# Pseudocode: KD-CNN training
teacher = TeacherCNN()    # 3-layer CNN
student = StudentCNN()    # 1-layer CNN

# Step 1: Train teacher on real labels
optimizer_T = Adam(teacher.parameters())
for x, y in train_loader:
    pred = teacher(x)
    lossT = CrossEntropyLoss(pred, y)
    lossT.backward(); optimizer_T.step(); optimizer_T.zero_grad()

# Step 2: Generate soft targets
soft_targets = []
for x, y in train_loader:
    probs = F.softmax(teacher(x), dim=1).detach()
    soft_targets.append(probs)
# Combine X with soft_targets for student training
soft_dataset = TensorDataset(X_all, concat(soft_targets))

# Step 3: Train student on teacher outputs
optimizer_S = Adam(student.parameters())
for x, soft_prob in soft_dataset:
    predS = student(x)
    lossS = -torch.sum(soft_prob * torch.log_softmax(predS, dim=1))
    lossS.backward(); optimizer_S.step(); optimizer_S.zero_grad()
```

In summary, KD-CNN attains near-teacher accuracy with a simpler model. It offers **fast inference** (simpler student forward pass) while preserving multi-class precision. The tradeoff is extra training complexity (teacher training + distillation), but this is acceptable for offline training.

&#x20;*Figure: A generic CNN architecture (from Liu et al. 2025) used as the teacher model. The student CNN uses a similar single conv→pool→softmax structure.*

### AC-GAN Approach for Synthetic Data and Robust Classification

*Auxiliary Classifier GANs* (AC-GANs) can generate synthetic hydroacoustic data and train classifiers robust to data scarcity. An AC-GAN consists of:

* **Generator (G):** Takes as input random noise $z$ and a class label $c$, and outputs a synthetic signal $G(z,c)$.
* **Discriminator (D):** Takes a signal (real or fake) and outputs two things: 1) probability $P(S=\text{real}|x)$ that the signal is real vs. fake, and 2) probability distribution $P(C|x)$ over class labels.

The AC-GAN is trained with the following objectives: the **source loss** $L_S$ and **class loss** $L_C$:

```math
L_S = \mathbb{E}_{x \sim p_{\rm real}}\bigl[\log P(S=\text{real}|x)\bigr] + \mathbb{E}_{z,c}\bigl[\log P(S=\text{fake}|G(z,c))\bigr],
```

```math
L_C = \mathbb{E}_{x \sim p_{\rm real}}\bigl[\log P(C=c|x)\bigr] + \mathbb{E}_{z,c}\bigl[\log P(C=c|G(z,c))\bigr],
```

where in $L_C$ the real signal $x$ has true class $c$, and $G(z,c)$ is conditioned on class $c$. The discriminator maximizes $L_S + L_C$ (correct source and class), while the generator maximizes $L_C - L_S$ (fool source and get class right).

After training converges, the **discriminator’s classification head** ($P(C|x)$) serves as a robust cavitation classifier. Importantly, since $D$ was trained to classify *real signals* into $N$ classes (cavitation states) and distinguish them from fake, one can *remove* the source output (real/fake) and use the class output for prediction on new signals. This yields a classifier that has learned from both real and generated examples, improving generalization.

**AC-GAN Benefits:** By generating *fake hydroacoustic signals* of each class, the AC-GAN augments the dataset where real samples are limited. Look *et al.* demonstrated in hydraulic turbines that an AC-GAN can push binary cavitation detection accuracy from \~80% (conventional CNN) to \~95%. Even when the generator collapsed, discriminator accuracy remained high (\~95–98%). The auxiliary classifier concept thus yields a *built-in* CNN trained on enriched data, making it robust to sensor location and turbine variations.

**Pseudocode:** AC-GAN training loop (simplified):

```python
# Pseudocode: AC-GAN training
generator = Generator()
discriminator = Discriminator()
opt_G = Adam(generator.parameters())
opt_D = Adam(discriminator.parameters())

for epoch in range(num_epochs):
    # --- Train Discriminator ---
    # 1. Real samples
    real_x, real_labels = sample_real_batch()
    D_real = discriminator(real_x)
    loss_D_real = -torch.mean(torch.log(D_real.source_real) + torch.log(D_real.class_prob[real_labels]))
    # 2. Fake samples
    z = torch.randn(batch_size, noise_dim)
    fake_labels = sample_labels(batch_size)
    fake_x = generator(z, fake_labels)
    D_fake = discriminator(fake_x.detach())
    loss_D_fake = -torch.mean(torch.log(1 - D_fake.source_real) + torch.log(D_fake.class_prob[fake_labels]))
    # Total discriminator loss
    loss_D = loss_D_real + loss_D_fake
    opt_D.zero_grad(); loss_D.backward(); opt_D.step()

    # --- Train Generator ---
    z = torch.randn(batch_size, noise_dim)
    gen_labels = sample_labels(batch_size)
    fake_x = generator(z, gen_labels)
    D_out = discriminator(fake_x)
    # Generator wants D to predict fake as real and correct class
    loss_G = -torch.mean(torch.log(D_out.source_real) + torch.log(D_out.class_prob[gen_labels]))
    opt_G.zero_grad(); loss_G.backward(); opt_G.step()
```

Here `D_out.source_real` is $P(S=\text{real}|x)$ and `D_out.class_prob[c]` is $P(C=c|x)$. After training, we discard $G$ and use $D$'s classification output as the cavitation detector. The **GAN losses** above correspond to maximizing $L_S+L_C$ for $D$ and $L_C - L_S$ for $G$.

### Comparison of CNN-Based Acoustic Classifiers

Different CNN-based methods trade off accuracy, speed, robustness, and complexity. A qualitative comparison is:

* **SSA-VMD-MSCNN:** Very high detection precision on nonstationary signals (superior to plain CNN or wavelet-CNN). Strong noise robustness due to adaptive decomposition. However, the VMD+SSA adds computational overhead (inner optimization) and many convolutional channels, making it relatively slow/inflexible for real-time. Complexity is high (SSA search + multiscale CNN). Scalability to new classes is straightforward by changing softmax.
* **1-D DHRN (multi-task):** Also high accuracy (94–100%) for both binary and multi-class cavitation. Its deep, hierarchical residual structure yields excellent generalization even in noisy data. Inference is moderate speed (18 layers) on 1-D inputs. Complexity is high (many filters, two shortcut paths) but scalability is good for adding output classes.
* **KD-CNN (Teacher–Student):** Student CNN is very fast at inference (single conv layer) with negligible latency, enabling near real-time detection. It achieves near-teacher accuracy (>97%). Robustness depends on teacher’s knowledge; it may be less able to handle unseen noise unless teacher was robust. Complexity: training is heavier (requires two networks), but deployment is simple. Scalability to new classes involves retraining teacher or student.
* **AC-GAN CNN:** The discriminator (used as CNN) shows high generalization; cavitation detection accuracy improved from \~80% (standard CNN) to \~95% by adversarial training. It is particularly robust to variations and limited data due to generated examples. However, inference speed is similar to a normal CNN, and training complexity is very high (GAN training). Scalability: naturally supports adding classes by adding labels to $G$ and $D$.

These qualitative assessments (summarized in the table below) are supported by reported results. For example, DHRN’s multi-task results (binary 97–100%, intensity 93–100%) and KD-CNN’s speed/accuracy (2s inference, >97% acc) illustrate the trade-offs above.

## B. Method Comparison

| Model                    | Accuracy (binary/multi)      | Inference Speed           | Noise Robustness                      | Complexity                | Scalability                          |
| ------------------------ | ---------------------------- | ------------------------- | ------------------------------------- | ------------------------- | ------------------------------------ |
| SSA-VMD-MSCNN            | Very high (e.g. >98% binary) | Low (heavy preprocessing) | High (adaptive VMD filters out noise) | High (SSA+VMD + deep CNN) | Easy (just adjust softmax)           |
| 1-D DHRN (multi-task)    | High (97–100% / 93–100%)     | Moderate (deep CNN)       | High (tested on noisy valves)         | High (18-layer residual)  | Good (multi-output learned together) |
| KD-CNN (Teacher→Student) | High (>97% multi-class)      | Very fast (1 conv layer)  | Moderate (depends on teacher)         | Low (student)             | Moderate (retrain for new classes)   |
| AC-GAN-based CNN         | High (\~95–98% binary)       | Moderate (CNN inference)  | Very high (GAN-trained generalizes)   | High (GAN training + CNN) | Good (can synthesize new classes)    |

The table illustrates that **KD-CNN** offers the fastest inference with little accuracy loss, making it appealing for real-time monitoring. **SSA-VMD-MSCNN** and **DHRN** excel in accuracy and noise robustness, at the expense of slower processing. **AC-GAN** enhances robustness via data augmentation, though it is mainly beneficial during training rather than online detection.

## C. Synthetic Data Generation

Data scarcity in cavitation monitoring is addressed by several augmentation and synthesis methods:

* **Sliding Window & FFT:** As in Sha et al., raw acoustic waveforms are sliced into overlapping windows and each window is transformed via FFT. This *Swin-FFT* approach multiplies the dataset size and introduces translation invariance. It also implicitly adds spectral perturbations, improving model robustness to slight frequency shifts.
* **GAN-Based Synthesis:** Auxiliary GANs (AC-GAN or other GAN variants) are used to generate synthetic hydroacoustic samples. For example, Look et al. used an AC-GAN to produce cavitation spectra for training a CNN. The generated samples fill gaps in the real dataset, especially for rare cavitation cases. GANs have also been used in related domains (e.g. cavitation detection in navy propellers) to improve classifier training.
* **Signal Transform Augmentation:** Techniques like adding Gaussian noise, time-stretching, or mixing samples (mixup) can augment acoustic signals. FFT jittering (randomizing phase), spectral shifting, or wavelet-based variations are used to create new examples. While no specific citation is given here, these methods are common practice in time-series augmentation literature.
* **Simulation of Cavitation Signals:** Physical models of cavitation (e.g. bubble dynamics) can generate data. Gatica *et al.* simulated acoustic cavitation by solving the Rayleigh–Plesset and related equations for bubble oscillation. They used these simulated signals (with labels “stable” vs “transient” cavitation) to train ML classifiers with high accuracy. Similarly, computational fluid dynamics (CFD) or multiphase flow models can provide synthetic pressure or vibration signals under known cavitation states. These physics-based data enrich training sets beyond what is feasible with experiments.
* **Label Noise and Mixing:** One can also inject label noise or mix different cavitation levels within a signal to generate intermediate cases. While not directly cited above, such techniques are used in fault diagnosis to expand class boundaries.

In summary, common augmentation strategies include window slicing, frequency-domain perturbations (FFT-based), GAN-generated samples, and simulated pressure/flow signals. These methods help prevent overfitting and improve generalization for cavitation classifiers.

## D. Conclusions

This report reviewed state-of-the-art neural methods for cavitation detection in hydraulic turbines, revealing trade-offs among approaches. **SSA-VMD-MSCNN** delivers high accuracy on nonstationary hydroacoustic data, thanks to adaptive multiscale filtering, but at the cost of computation. **1-D DHRN** networks achieve both binary cavitation detection and multi-level intensity classification with very high precision, especially when combined with multi-task learning and frequency-domain augmentation. **Knowledge-distillation CNNs** provide a practical real-time solution: a large teacher CNN imparts its knowledge to a small student, yielding fast inference (\~1-layer CNN) while retaining >97% accuracy. **AC-GAN** methods enhance robustness and data efficiency by synthesizing realistic cavitation signals, boosting classifier generalization. Lastly, advanced signal processing (e.g. envelope analysis) complements CNNs: envelope spectra and variance metrics can pinpoint cavitation-induced frequencies.

For a real-time implementation, we recommend a hybrid pipeline: first preprocess raw hydroacoustic signals with optimized VMD (SSA-VMD) to handle nonstationarity and noise, then input the decomposed components into a lightweight CNN. The CNN could be a student model distilled from a larger teacher or a specialized DHRN, depending on hardware constraints. Training should leverage extensive data augmentation – combining sliding-window FFT augmentation and GAN-generated samples – to cover possible flow conditions. For binary cavitation alarm, the KD-CNN student affords minimal latency. For more granular diagnosis (cavitation levels), a small DHRN-like branch or multi-head softmax can be added. In practice, this could look like: perform VMD → feed IMFs to a 2-channel CNN (one channel per IMF subset) → run through a shallow CNN (fast inference) → output cavitation/no-cavitation. Periodically retrain or fine-tune the model with newly collected data (or simulated data) to maintain performance.

In conclusion, blending adaptive signal decomposition (SSA-VMD), modern CNN architectures (residual or distilled), and advanced augmentation yields a powerful, robust cavitation monitoring system suitable for on-line turbine health management.

## E. Bibliography

\[1] F. Li, C. Wang, H. Li, Z. Liu, Y. Huang, and T. Wang, “Sparrow search algorithm-optimized variational mode decomposition-based multiscale convolutional network for cavitation diagnosis of hydro turbines,” *Ocean Eng.*, vol. 312, p. 119055, 2024.

\[2] Y. Sha, J. Faber, S. Gou, B. Liu, W. Li, S. Schramm, H. Stoecker, T. Steckenreiter, D. Vnucec, N. Wetzstein, A. Widl, and K. Zhou, “A multi-task learning framework for simultaneous cavitation detection and cavitation intensity recognition of valve acoustic signals using 1-D double hierarchical residual networks,” *Engineering Applications of Artificial Intelligence*, vol. 117, 105226, 2023.

\[3] Z. Liu, Z. Zhou, S. Zou, Z. Liu, and S. Qiao, “Cavitation state identification method of hydraulic turbine based on knowledge distillation and convolutional neural network,” *Power Generation Technology*, vol. 46, no. 1, pp. 161–170, 2025.

\[4] A. Look, O. Kirschner, and S. Riedelbauch, “Building robust classifiers with generative adversarial networks for detecting cavitation in hydraulic turbines,” in *Proc. 7th Int. Conf. Pattern Recognit. Appl. Methods (ICPRAM 2018)*, pp. 454–460, 2018.

\[5] H. Liu, Z. Tong, B. Shang, et al., “Cavitation diagnostics based on self-tuning variational mode decomposition for fluid machinery with low-SNR conditions,” *Chin. J. Mech. Eng.*, vol. 36, no. 1, 2023.

\[6] T. Gatica, E. van ’t Wout, and R. Haqshenas, “Classifying acoustic cavitation with machine learning trained on multiple physical models,” *arXiv:2408.16142*, 2024.
