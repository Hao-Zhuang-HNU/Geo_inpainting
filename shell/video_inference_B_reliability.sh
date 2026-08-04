#mask_20_30
python ../video_inference.py \
   -i /root/autodl-tmp/Inpainting/BundleFusion/B_fault/B_imgs_20_30/office0/ \
  --mask /root/autodl-tmp/Inpainting/masks/irregular_mask/mask_rates_20_30 \
  --line /root/autodl-tmp/Inpainting/Geo_inpainting/geo_output/B_mask_20_30/office0/line/ \
  --ckpt_path ../experiments_model/propainter_train_LinePro_robust/gen_050000.pth \
  --width 256 --height 256 \
  --line_soft_guidance \
  -o ../results/B_mask_20_30_reliability/office0/ \
  --save_frames --save_line_debug
python ../video_inference.py \
   -i /root/autodl-tmp/Inpainting/BundleFusion/B_fault/B_imgs_20_30/office1/ \
  --mask /root/autodl-tmp/Inpainting/masks/irregular_mask/mask_rates_20_30 \
  --line /root/autodl-tmp/Inpainting/Geo_inpainting/geo_output/B_mask_20_30/office1/line/ \
  --ckpt_path ../experiments_model/propainter_train_LinePro_robust/gen_050000.pth \
  --width 256 --height 256 \
  --line_soft_guidance \
  -o ../results/B_mask_20_30_reliability/office1/ \
  --save_frames --save_line_debug
python ../video_inference.py \
   -i /root/autodl-tmp/Inpainting/BundleFusion/B_fault/B_imgs_20_30/office2/ \
  --mask /root/autodl-tmp/Inpainting/masks/irregular_mask/mask_rates_20_30 \
  --line /root/autodl-tmp/Inpainting/Geo_inpainting/geo_output/B_mask_20_30/office2/line/ \
  --ckpt_path ../experiments_model/propainter_train_LinePro_robust/gen_050000.pth \
  --width 256 --height 256 \
  --line_soft_guidance \
  -o ../results/B_mask_20_30_reliability/office2/ \
  --save_frames --save_line_debug
python ../video_inference.py \
   -i /root/autodl-tmp/Inpainting/BundleFusion/B_fault/B_imgs_20_30/office3/ \
  --mask /root/autodl-tmp/Inpainting/masks/irregular_mask/mask_rates_20_30 \
  --line /root/autodl-tmp/Inpainting/Geo_inpainting/geo_output/B_mask_20_30/office3/line/ \
  --ckpt_path ../experiments_model/propainter_train_LinePro_robust/gen_050000.pth \
  --width 256 --height 256 \
  --line_soft_guidance \
  -o ../results/B_mask_20_30_reliability/office3/ \
  --save_frames --save_line_debug




#mask_10_20
python ../video_inference.py \
   -i /root/autodl-tmp/Inpainting/BundleFusion/B_fault/B_imgs_10_20/office0/ \
  --mask /root/autodl-tmp/Inpainting/masks/irregular_mask/mask_rates_10_20 \
  --line /root/autodl-tmp/Inpainting/Geo_inpainting/geo_output/B_mask_10_20/office0/line/ \
  --ckpt_path ../experiments_model/propainter_train_LinePro_robust/gen_050000.pth \
  --width 256 --height 256 \
  --line_soft_guidance \
  -o ../results/B_mask_10_20_reliability/office0/ \
  --save_frames --save_line_debug
python ../video_inference.py \
   -i /root/autodl-tmp/Inpainting/BundleFusion/B_fault/B_imgs_10_20/office1/ \
  --mask /root/autodl-tmp/Inpainting/masks/irregular_mask/mask_rates_10_20 \
  --line /root/autodl-tmp/Inpainting/Geo_inpainting/geo_output/B_mask_10_20/office1/line/ \
  --ckpt_path ../experiments_model/propainter_train_LinePro_robust/gen_050000.pth \
  --width 256 --height 256 \
  --line_soft_guidance \
  -o ../results/B_mask_10_20_reliability/office1/ \
  --save_frames --save_line_debug
python ../video_inference.py \
   -i /root/autodl-tmp/Inpainting/BundleFusion/B_fault/B_imgs_10_20/office2/ \
  --mask /root/autodl-tmp/Inpainting/masks/irregular_mask/mask_rates_10_20 \
  --line /root/autodl-tmp/Inpainting/Geo_inpainting/geo_output/B_mask_10_20/office2/line/ \
  --ckpt_path ../experiments_model/propainter_train_LinePro_robust/gen_050000.pth \
  --width 256 --height 256 \
  --line_soft_guidance \
  -o ../results/B_mask_10_20_reliability/office2/ \
  --save_frames --save_line_debug
python ../video_inference.py \
   -i /root/autodl-tmp/Inpainting/BundleFusion/B_fault/B_imgs_10_20/office3/ \
  --mask /root/autodl-tmp/Inpainting/masks/irregular_mask/mask_rates_10_20 \
  --line /root/autodl-tmp/Inpainting/Geo_inpainting/geo_output/B_mask_10_20/office3/line/ \
  --ckpt_path ../experiments_model/propainter_train_LinePro_robust/gen_050000.pth \
  --width 256 --height 256 \
  --line_soft_guidance \
  -o ../results/B_mask_10_20_reliability/office3/ \
  --save_frames --save_line_debug
  
  
  
  
  
#mask_5_10
python ../video_inference.py \
   -i /root/autodl-tmp/Inpainting/BundleFusion/B_fault/B_imgs_5_10/office0/ \
  --mask /root/autodl-tmp/Inpainting/masks/irregular_mask/mask_rates_5_10 \
  --line /root/autodl-tmp/Inpainting/Geo_inpainting/geo_output/B_mask_5_10/office0/line/ \
  --ckpt_path ../experiments_model/propainter_train_LinePro_robust/gen_050000.pth \
  --width 256 --height 256 \
  --line_soft_guidance \
  -o ../results/B_mask_5_10_reliability/office0/ \
  --save_frames --save_line_debug
python ../video_inference.py \
   -i /root/autodl-tmp/Inpainting/BundleFusion/B_fault/B_imgs_5_10/office1/ \
  --mask /root/autodl-tmp/Inpainting/masks/irregular_mask/mask_rates_5_10 \
  --line /root/autodl-tmp/Inpainting/Geo_inpainting/geo_output/B_mask_5_10/office1/line/ \
  --ckpt_path ../experiments_model/propainter_train_LinePro_robust/gen_050000.pth \
  --width 256 --height 256 \
  --line_soft_guidance \
  -o ../results/B_mask_5_10_reliability/office1/ \
  --save_frames --save_line_debug
python ../video_inference.py \
   -i /root/autodl-tmp/Inpainting/BundleFusion/B_fault/B_imgs_5_10/office2/ \
  --mask /root/autodl-tmp/Inpainting/masks/irregular_mask/mask_rates_5_10 \
  --line /root/autodl-tmp/Inpainting/Geo_inpainting/geo_output/B_mask_5_10/office2/line/ \
  --ckpt_path ../experiments_model/propainter_train_LinePro_robust/gen_050000.pth \
  --width 256 --height 256 \
  --line_soft_guidance \
  -o ../results/B_mask_5_10_reliability/office2/ \
  --save_frames --save_line_debug
python ../video_inference.py \
   -i /root/autodl-tmp/Inpainting/BundleFusion/B_fault/B_imgs_5_10/office3/ \
  --mask /root/autodl-tmp/Inpainting/masks/irregular_mask/mask_rates_5_10 \
  --line /root/autodl-tmp/Inpainting/Geo_inpainting/geo_output/B_mask_5_10/office3/line/ \
  --ckpt_path ../experiments_model/propainter_train_LinePro_robust/gen_050000.pth \
  --width 256 --height 256 \
  --line_soft_guidance \
  -o ../results/B_mask_5_10_reliability/office3/ \
  --save_frames --save_line_debug
  
  
  
  
  
  
  
  
  
  
  
  
  
# ##mask_30_40
# python ../video_inference.py \
# -i /root/autodl-tmp/Inpainting/BundleFusion/B_fault/B_imgs_30_40/office0/ \
# --mask /root/autodl-tmp/Inpainting/masks/irregular_mask/mask_rates_30_40 \
# --line /root/autodl-tmp/Inpainting/Geo_inpainting/geo_output/B_mask_30_40/office0/line/ \
# --ckpt_path ../experiments_model/propainter_train_LinePro_robust/gen_050000.pth \
# --width 256 --height 256 \
# --line_soft_guidance \
# -o ../results/B_mask_30_40_reliability/office0/ \
# --save_frames --save_line_debug
python ../video_inference.py \
-i /root/autodl-tmp/Inpainting/BundleFusion/B_fault/B_imgs_30_40/office1/ \
--mask /root/autodl-tmp/Inpainting/masks/irregular_mask/mask_rates_30_40 \
--line /root/autodl-tmp/Inpainting/Geo_inpainting/geo_output/B_mask_30_40/office1/line/ \
--ckpt_path ../experiments_model/propainter_train_LinePro_robust/gen_050000.pth \
--width 256 --height 256 \
--line_soft_guidance \
-o ../results/B_mask_30_40_reliability/office1/ \
--save_frames --save_line_debug
# python ../video_inference.py \
# -i /root/autodl-tmp/Inpainting/BundleFusion/B_fault/B_imgs_30_40/office2/ \
# --mask /root/autodl-tmp/Inpainting/masks/irregular_mask/mask_rates_30_40 \
# --line /root/autodl-tmp/Inpainting/Geo_inpainting/geo_output/B_mask_30_40/office2/line/ \
# --ckpt_path ../experiments_model/propainter_train_LinePro_robust/gen_050000.pth \
# --width 256 --height 256 \
# --line_soft_guidance \
# -o ../results/B_mask_30_40_reliability/office2/ \
# --save_frames --save_line_debug
# python ../video_inference.py \
# -i /root/autodl-tmp/Inpainting/BundleFusion/B_fault/B_imgs_30_40/office3/ \
# --mask /root/autodl-tmp/Inpainting/masks/irregular_mask/mask_rates_30_40 \
# --line /root/autodl-tmp/Inpainting/Geo_inpainting/geo_output/B_mask_30_40/office3/line/ \
# --ckpt_path ../experiments_model/propainter_train_LinePro_robust/gen_050000.pth \
# --width 256 --height 256 \
# --line_soft_guidance \
# -o ../results/B_mask_30_40_reliability/office3/ \
# --save_frames --save_line_debug




# ##mask_40_50
# python ../video_inference.py \
# -i /root/autodl-tmp/Inpainting/BundleFusion/B_fault/B_imgs_40_50/office0/ \
# --mask /root/autodl-tmp/Inpainting/masks/irregular_mask/mask_rates_40_50 \
# --line /root/autodl-tmp/Inpainting/Geo_inpainting/geo_output/B_mask_40_50/office0/line/ \
# --ckpt_path ../experiments_model/propainter_train_LinePro_robust/gen_050000.pth \
# --width 256 --height 256 \
# --line_soft_guidance \
# -o ../results/B_mask_40_50_reliability/office0/ \
# --save_frames --save_line_debug
python ../video_inference.py \
-i /root/autodl-tmp/Inpainting/BundleFusion/B_fault/B_imgs_40_50/office1/ \
--mask /root/autodl-tmp/Inpainting/masks/irregular_mask/mask_rates_40_50 \
--line /root/autodl-tmp/Inpainting/Geo_inpainting/geo_output/B_mask_40_50/office1/line/ \
--ckpt_path ../experiments_model/propainter_train_LinePro_robust/gen_050000.pth \
--width 256 --height 256 \
--line_soft_guidance \
-o ../results/B_mask_40_50_reliability/office1/ \
--save_frames --save_line_debug
# python ../video_inference.py \
# -i /root/autodl-tmp/Inpainting/BundleFusion/B_fault/B_imgs_40_50/office2/ \
# --mask /root/autodl-tmp/Inpainting/masks/irregular_mask/mask_rates_40_50 \
# --line /root/autodl-tmp/Inpainting/Geo_inpainting/geo_output/B_mask_40_50/office2/line/ \
# --ckpt_path ../experiments_model/propainter_train_LinePro_robust/gen_050000.pth \
# --width 256 --height 256 \
# --line_soft_guidance \
# -o ../results/B_mask_40_50_reliability/office2/ \
# --save_frames --save_line_debug
# python ../video_inference.py \
# -i /root/autodl-tmp/Inpainting/BundleFusion/B_fault/B_imgs_40_50/office3/ \
# --mask /root/autodl-tmp/Inpainting/masks/irregular_mask/mask_rates_40_50 \
# --line /root/autodl-tmp/Inpainting/Geo_inpainting/geo_output/B_mask_40_50/office3/line/ \
# --ckpt_path ../experiments_model/propainter_train_LinePro_robust/gen_050000.pth \
# --width 256 --height 256 \
# --line_soft_guidance \
# -o ../results/B_mask_40_50_reliability/office3/ \
# --save_frames --save_line_debug
