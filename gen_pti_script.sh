#!/usr/bin/env bash

models=("your model name")

in="models"
out="pti_out"

for model in ${models[@]}

do

    for i in 0 1

    do 
        python projector_withseg.py --outdir=${out} --target_img=dataset/testdata_img --network ${in}/${model} --idx ${i}

        python gen_videos_proj_withseg.py --output=${out}/${model}/${i}/PTI_render/post.mp4 --latent=${out}/${model}/${i}/projected_w.npz --trunc 0.7 --network ${out}/${model}/${i}/fintuned_generator.pkl --cfg Head
    done

done
