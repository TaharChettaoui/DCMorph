# DCMorph: Face Morphing via Dual-Stream Cross-Attention Diffusion

Tahar Chettaoui, 
Eduarda Caldeira, 
Guray Ozgur, 
Raghavendra Ramachandra, 
Fadi Boutros, 
Naser Damer

[[`Paper`](https://openaccess.thecvf.com/content/CVPR2026W/BIOM2026/html/Chettaoui_DCMorph_Face_Morphing_via_Dual-Stream_Cross-Attention_Diffusion_CVPRW_2026_paper.html)] 

### Abstract
Advancing face morphing attack techniques is crucial to anticipate evolving threats and develop robust defensive mechanisms for identity verification systems. This work introduces DCMorph, a dual-stream diffusion-based morphing framework that simultaneously operates at both identity conditioning and latent space levels. Unlike imagelevel methods suffering from blending artifacts or GANbased approaches with limited reconstruction fidelity, DCMorph leverages identity-conditioned latent diffusion models through two mechanisms: (1) decoupled cross-attention interpolation that injects identity-specific features from both source faces into the denoising process, enabling explicit dual-identity conditioning absent in existing diffusionbased methods, and (2) DDIM inversion with spherical interpolation between inverted latent representations from both source faces, providing geometrically consistent initial latent representation that preserves structural attributes. Vulnerability analyses across four state-of-the-art face recognition systems demonstrate that DCMorph achieves the highest attack success rates compared to existing methods at both operational thresholds, while remaining challenging to detect by current morphing attack detection solutions.

### Citation

```
@InProceedings{Chettaoui_2026_CVPR,
    author    = {Chettaoui, Tahar and Caldeira, Eduarda and Ozgur, Guray and Ramachandra, Raghavendra and Boutros, Fadi and Damer, Naser},
    title     = {DCMorph: Face Morphing via Dual-Stream Cross-Attention Diffusion},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2026},
    pages     = {1556-1566}
}
```

### License 

```
This project is licensed under the terms of the Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license. 
Copyright (c) 2021 Fraunhofer Institute for Computer Graphics Research IGD Darmstadt
```
