# LLM4UATR.
Underwater acoustic target recognition (UATR) is essential for maritime surveillance and ocean engineering, yet it suffers from weak, nonstationary physical signals and a critical scarcity of labeled data. Current deep learning-based approaches typically rely on acoustics-only features, overlooking the rich, structured semantic metadata (e.g., vessel specifications and states) available in maritime scenarios. 
Furthermore, due to the limited scale and lack
of diversity in labeled underwater acoustic data, transferring general-purpose audio foundation models to UATR often encounters severe domain mismatch issues caused by complex underwater propagation channels.
To address these challenges, this paper proposes model, a Large Language Model (LLM)-empowered multimodal framework for UATR via cross-modality alignment (CMA). 
Instead of conventional classification, model formulates UATR as semantic-acoustic joint representation learning. It features an audio adaptation encoder that pairs a parameter-efficient adapter with a bidirectional Mamba module to mitigate domain mismatch and capture long-range, ship-specific temporal acoustic patterns. 
Simultaneously, a text branch based on GPT-2 extracts LLM-derived semantic priors from vessel metadata. A cross-modality alignment mechanism is developed to leverage these textual representations to guide acoustic feature aggregation in a shared embedding space. Extensive experiments conducted on three benchmark datasets—ShipsEar, DeepShip, and Oceanship—demonstrate that model achieves state-of-the-art recognition performance and exhibits superior robustness under noisy, data-sparse conditions and variations in segment duration. 

In summary, our work makes the following three key contributions:
• LLM-guided semantic priors are introduced for underwater acoustic target recognition.
• BEATs adaptation and bidirectional Mamba modeling improve acoustic feature learning.
• Cross-modality alignment enhances robustness across datasets and segment durations
