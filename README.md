# FindTrack

This is the official PyTorch implementation of our paper:

> **Find First, Track Next: Decoupling Identification and Propagation in Referring Video Object Segmentation**, *ICCVW 2025*\
> Suhwan Cho*, Seunghoon Lee*, Minhyeok Lee, Jungho Lee, Sangyoun Lee\
> Link: [[arXiv]](https://arxiv.org/abs/2503.03492)

<img src="https://github.com/user-attachments/assets/a57cd78a-6f34-4fa7-bddc-762e5e90a71b" width=800>

You can also explore other related works at [awesome-video-object segmentation](https://github.com/suhwan-cho/awesome-video-object-segmentation).


## Demo Video

https://github.com/user-attachments/assets/e5475442-f2fe-4899-84dd-8ae7ef22a7f2


## Abstract
Existing referring VOS methods typically fuse visual and textual features in a highly entangled manner, processing multi-modal information jointly. 
However, this entanglement often leads to challenges in resolving ambiguous target identification and maintaining consistent mask propagation across frames.
To address these issues, we propose **a decoupled framework** that explicitly separates object identification from mask propagation. 
The key frame is adaptively selected based on segmentation confidence and vision-text alignment, establishing **a reliable anchor for propagation**.


## Setup
1\. Download the datasets:
[Ref-YouTube-VOS](https://codalab.lisn.upsaclay.fr/competitions/3282),
[Ref-DAVIS17](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/video-segmentation/video-object-segmentation-with-language-referring-expressions),
[MeViS](https://codalab.lisn.upsaclay.fr/competitions/15094).


2\. Download [Alpha-CLIP](https://drive.google.com/file/d/1dG_j98hh7AFvhSADlhp9CpoNY-9rBHoc/view?usp=drive_link) weights and place it in the ``weights/`` directory.


## Running 


### Training (optional)
FindTrack works well in a training-free manner, but fine-tuning on specific datasets can improve performance further.

For Ref-YouTube-VOS dataset:
```
deepspeed --num_gpus 4 train_ytvos.py 
```

For MeViS dataset:
```
deepspeed --num_gpus 4 train_mevis.py 
```


### Testing
For Ref-YouTube-VOS dataset:
```
python run_ytvos.py
```

For MeViS dataset:
```
python run_mevis.py
```

Verify the following before running:\
✅ Testing dataset selection\
✅ GPU availability and configuration\
✅ Pre-trained model path


### Gradio Demo
You can use the web demo with your own video!

<img src="https://github.com/user-attachments/assets/74eb0778-84dd-4f84-b081-3bfae8de91d7" width=600>

Run the Gradio demo with:
```
python demo.py
```


## Attachments
[Pre-computed results](https://drive.google.com/file/d/1rhk3gWbuUem3-XvtlFJG-SyehNjX3zL_/view?usp=drive_link)


## Contact
Code and models are only available for non-commercial research purposes.\
For questions or inquiries, feel free to contact:
```
E-mail: suhwanx@gmail.com
```
