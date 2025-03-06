# FindTrack

This is the official PyTorch implementation of our paper:

> **Find First, Track Next: Decoupling Identification and Propagation in Referring Video Object Segmentation**\
> Suhwan Cho*, Seunghoon Lee*, Minhyeok Lee, Jungho Lee, Sangyoun Lee\
> Link: [[arXiv]](https://arxiv.org/abs/2503.03492)

<img src="https://github.com/user-attachments/assets/a57cd78a-6f34-4fa7-bddc-762e5e90a71b" width=800>

You can also find other related papers at [awesome-video-object segmentation](https://github.com/suhwan-cho/awesome-video-object-segmentation).


## Demo Video

https://github.com/user-attachments/assets/e5475442-f2fe-4899-84dd-8ae7ef22a7f2

## Abstract
Existing referring VOS methods typically fuse visual and textual features in a highly entangled manner, processing multi-modal information jointly. 
However, this entanglement often leads to challenges in resolving ambiguous target identification and maintaining consistent mask propagation across frames.
To address these issues, we propose **a decoupled framework** that explicitly separates object identification from mask propagation. 
The key frame is adaptively selected based on segmentation confidence and vision-text alignment, establishing **a reliable anchor for propagation**.
