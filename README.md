<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;">
                Few-shot Image Generation via Adaptation-Aware <br> Kernel Modulation</h1>
<p align='center' style="text-align:center;font-size:1.25em;">
    <a href="https://yunqing-me.github.io/" target="_blank" style="text-decoration: none;">Yunqing Zhao*</a>&nbsp;/&nbsp;
    <a href="https://keshik6.github.io/" target="_blank" style="text-decoration: none;">Keshigeyan Chandrasegaran*</a>&nbsp;/&nbsp;
    <a href="https://miladabd.github.io/" target="_blank" style="text-decoration: none;">Milad Abdollahzadeh*</a>&nbsp;/&nbsp;
    <a href="https://sites.google.com/site/mancheung0407/" target="_blank" style="text-decoration: none;">Ngai&#8209;Man Cheung</a></br>
Singapore University of Technology and Design (<b>SUTD</b>)<br/>
</p>

<p align='center';>
<b>
<em>The Thirty-Sixth Annual Conference on Neural Information Processing Systems (NeurIPS 2022);</em> <br>
<em>Ernest N. Morial Convention Center, New Orleans, LA, USA.</em>
</b>
</p>

<p align='center' style="text-align:center;font-size:2.5 em;">
<b>
    <a href="https://yunqing-me.github.io/FSIG-ImportanceProbing-KML/" target="_blank" style="text-decoration: none;">Project Page</a>&nbsp;/&nbsp;
    <a href="https://yunqing-me.github.io/FSIG-ImportanceProbing-KML/" target="_blank" style="text-decoration: none;">Poster</a>&nbsp;/&nbsp;
    <a href="https://yunqing-me.github.io/FSIG-ImportanceProbing-KML/" target="_blank" style="text-decoration: none;">Paper</a>&nbsp;/&nbsp;
    <a href="https://yunqing-me.github.io/FSIG-ImportanceProbing-KML/" target="_blank" style="text-decoration: none;">Talk</a>&nbsp;
</b>
</p>


----------------------------------------------------------------------

#### TL, DR: 
```
In this research, the proposed method aims to identify kernels in source GAN important for few-shot target adaptation and protect them from distortion. 

After this Importance Probing stage, the model can then perform few-shot adaptation using very few samples from target domains with different proximity.
```

## Installation and Environment:

- Platform: Linux
- Tesla V100 GPUs with CuDNN 10.1
- PyTorch 1.7.0
- Python 3.6.9
- lmdb, tqdm

Alternatively, you can install all libiraries through:  `pip install -r requirements.txt`

## Analysis of Source ↦ Target distance

We analyze the Source ↦ Target domain relation in the Sec. 3 (and Supplementary). See below for related steps in this analysis.

Step 1. `git clone https://github.com/rosinality/stylegan2-pytorch.git`

Step 2. Move `./visualization` to `./stylegan2-pytorch`

Step 3. Then, refer to the visualization code in `./visualization`.

## Pre-processing for training

### Step 1. 
Prepare the few-shot training dataset using lmdb format

For example, download the 10-shot target set, `Babies` ([Link](https://drive.google.com/file/d/1P8JMLq2Kk61MbEZDgwytqXxfrhG-NqcR/view?usp=sharing)) and `AFHQ-Cat`([Link](https://drive.google.com/file/d/1zgacEE0jiiDxttbK81fk6miY_4Ithhw-/view?usp=sharing)), and organize your directory as follows:

~~~
10-shot-{babies/afhq_cat}
└── images		
    └── image-1.png
    └── image-2.png
    └── ...
    └── image-10.png
~~~

Then, transform to lmdb format:

`python prepare_data.py --input_path [your_data_path_of_{babies/afhq_cat}] --output_path [your_lmdb_data_path_of_{babies/afhq_cat}]`

### Step 2. 
Prepare the entire target dataset for evaluation

For example, download the entire dataset, `Babies`([Link](https://drive.google.com/file/d/1JmjKBq_wylJmpCQ2OWNMy211NFJhHHID/view?usp=sharing)) and `AFHQ-Cat`([Link](https://github.com/clovaai/stargan-v2/blob/master/README.md#animal-faces-hq-dataset-afhq)), and organize your directory as follows:

~~~
entire-{babies/afhq_cat}
└── images		
    └── image-1.png
    └── image-2.png
    └── ...
    └── image-n.png
~~~

Then, transform to lmdb format for evaluation

`python prepare_data.py --input_path [your_data_path_of_entire_{babies/afhq_cat}] --output_path [your_lmdb_data_path_of_entire_{babies/afhq_cat}]`

### Step 3. 
Download the GAN model pretrained on FFHQ from [here](https://drive.google.com/file/d/1TQ_6x74RPQf03mSjtqUijM4MZEMyn7HI/view). Then, save it to `./_pretrained/style_gan_source_ffhq.pt`.

## Experiments

## Training & Evaluation: 

### Step 1. Adaptation-aware Importance Probing (IP) to indentify important kernels for target domain

~~~bash
# The probing process is lightweight and will only consume ~8mins
CUDA_VISIBLE_DEVICES=7 python _low_rank_probing.py \
 --exp _low_rank_probing_ffhq-{babies/afhq_cat}_10 --data_path {babies/afhq_cat} --n_sample_train 10 \
 --fisher_iter 500 --fisher_freq 500 --num_batch_fisher 250 --source_key ffhq --ckpt_source style_gan_source_ffhq.pt \
 --wandb --wandb_project_name test_probing-ffhq-{babies/afhq_cat} --wandb_run_name EstFisher 
~~~

We can obtain the estimated Fisher information of modulated kernels and it will be saved in `./_output_style_gan/args.exp/checkpoints/filter_fisher_g.pt` and `./_output_style_gan/args.exp/checkpoints/filter_fisher_d.pt`

10-shot Target Images, Estimated Fisher Information and Weights can be found [Here](https://drive.google.com/drive/folders/1cLA134v7aOOt6lh_faqd6WoqOyCx1Etk?usp=sharing)

## Step 2.  Adaptation-aware Kernel Modulation for Few-shot Image generation (FSIG)

~~~bash
 # The adaptation process is also computational efficient, it will lasts ~65mins for Babies and ~110 mins for AFHQ-Cat.
 CUDA_VISIBLE_DEVICES=7 python train_filter_kml.py \
  --exp filter_kml_ffhq-{babies/afhq_cat}_10 --data_path {babies/afhq_cat} --n_sample_train 10 \
  --eval_in_training --eval_in_training_freq 100 --iter {1500/3000} \
  --wandb --wandb_project_name filter_kml-{babies/afhq_cat} --wandb_run_name filter_kml --batch 4 --n_sample_test 5000 \
  --store_samples --samples_freq 100 \ # this can generate intermediate images during training
~~~

Training dynamics and evaluation results will be shown on [`wandb`](https://wandb.ai/site)

We note that, ideally Step 1. and Step 2. can be combined together. Here, for simplicity we use two steps as demonstration.

## Evaluate our method on more GAN architectures

To evaluate the effectiveness of our method, we further provide the implementation of ProGAN. Please find below: `https://drive.google.com/drive/folders/1aVfKnUIRKmFHODGeF4vvcNeMZC7H5E8A?usp=sharing`, which includs the Importance Probing and Adapataion implementations.

## Visualize the Important Kernels identified in Probing Stage.

We use the official implementation (`https://github.com/CSAILVision/GANDissect`) to visualize the important kernels via GAN dissection. Please check the supplementary for more details.

## Training your own GAN !

We provide all 10-shot target images and models used in our main paper and Supplementary. You can also adapt to other images determined by you.

Source GAN:
- [FFHQ](https://drive.google.com/file/d/1TQ_6x74RPQf03mSjtqUijM4MZEMyn7HI/view)
- [LSUN-Church](https://drive.google.com/file/d/18NlBBI8a61aGBHA1Tr06DQYlf-DRrBOH/view)
- [LSUN-Cars](https://drive.google.com/file/d/1O-yWYNvuMmirN8Q0Z4meYoSDtBfJEjGc/view)
- ...

Target Samples: [Link](https://drive.google.com/drive/folders/10skBzKjr8jJbWvTXKgA0yj-gT-aojRIE?usp=sharing)

- Babies
- Sunglasses
- MetFaces
- AFHQ-Cat
- AFHQ-Dog
- AFHQ-Wild
- Sketches
- Amedeo Modigliani's Paintings
- Rafael's Paintings
- Otto Dix's Paintings
- Haunted houses
- Van Gogh houses
- Wrecked cars
- ...

Follow Experiments part in this repo and you can produce your customized results.

## Bibtex

```
@inproceedings{zhao2022fsig-ip,
  title={Few-shot Image Generation via Adaptation-Aware Kernel Modulation},
  author={Zhao, Yunqing and Chandrasegaran, Keshigeyan and Abdollahzadeh, Milad and Cheung, Ngai-Man},
  booktitle={Neurips},
  year={2022}
}
```

## Acknowledgement: 

We appriciate the wonderful base implementation of StyleGAN V2 implementation from [@rosinality](https://github.com/rosinality). We also thank  [@mseitzer](https://github.com/mseitzer/pytorch-fid), [@Ojha](https://github.com/utkarshojha/few-shot-gan-adaptation) and [@richzhang](https://github.com/richzhang/PerceptualSimilarity) for their implementations on FID score and intra-LPIPS.

We also thank for the useful training and evaluation tool used in this work, from [@Miaoyun](https://github.com/MiaoyunZhao/GANmemory_LifelongLearning).



