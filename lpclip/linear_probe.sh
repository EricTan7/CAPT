# feature_dir=clip_feat
feature_dir=/mnt/sdc/tanhao/Baseline_v1/lpclip/clip_feat

# UCF101 Food101 StanfordCars EuroSAT DescribableTextures FGVCAircraft OxfordFlowers OxfordPets ImageNet
for DATASET in ImageNet
do
    python linear_probe.py \
    --dataset ${DATASET} \
    --feature_dir ${feature_dir} \
    --num_step 8 \
    --num_run 3
done
