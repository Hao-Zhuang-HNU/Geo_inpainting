# ##5_10
python ../video_inference.py \
--ckpt_path ../experiments_model/propainter_train_LinePro_robust/gen_050000.pth \
-m /root/autodl-tmp/Inpainting/masks/irregular_mask/mask_rates_5_10 \
--save_frames -i /root/autodl-tmp/Inpainting/BundleFusion/B_fault/B_imgs_5_10/office0 \
--line ../geo_output/B_office0_5_10_02/line/ \
-o ../results/B_office0_5_10 --width 256 --height 256

python ../video_inference.py \
--ckpt_path ../experiments_model/propainter_train_LinePro_laplacianloss/gen_040000.pth \
-m /root/autodl-tmp/Inpainting/masks/irregular_mask/mask_rates_5_10 \
--save_frames -i /root/autodl-tmp/Inpainting/BundleFusion/B_fault/B_imgs_5_10/office1 \
--line ../geo_output/B_office1_5_10_02/line/ \
-o ../results/B_office1_5_10 --width 256 --height 256

python ../video_inference.py \
--ckpt_path ../experiments_model/propainter_train_LinePro_laplacianloss/gen_040000.pth \
-m /root/autodl-tmp/Inpainting/masks/irregular_mask/mask_rates_5_10 \
--save_frames -i /root/autodl-tmp/Inpainting/BundleFusion/B_fault/B_imgs_5_10/office2 \
--line ../geo_output/B_office2_5_10_02/line/ \
-o ../results/B_office2_5_10 --width 256 --height 256

python ../video_inference.py \
--ckpt_path ../experiments_model/propainter_train_LinePro_laplacianloss/gen_040000.pth \
-m /root/autodl-tmp/Inpainting/masks/irregular_mask/mask_rates_5_10 \
--save_frames -i /root/autodl-tmp/Inpainting/BundleFusion/B_fault/B_imgs_5_10/office3 \
--line ../geo_output/B_office3_5_10_02/line/ \
-o ../results/B_office3_5_10 --width 256 --height 256















##10_20
python ../video_inference.py \
--ckpt_path ../experiments_model/propainter_train_LinePro_laplacianloss/gen_040000.pth \
-m /root/autodl-tmp/Inpainting/masks/irregular_mask/mask_rates_10_20 \
--save_frames -i /root/autodl-tmp/Inpainting/BundleFusion/B_fault/B_imgs_10_20/office0 \
--line ../geo_output/B_office0_10_20_02/line/ \
-o ../results/B_office0_10_20 --width 256 --height 256

python ../video_inference.py \
--ckpt_path ../experiments_model/propainter_train_LinePro_laplacianloss/gen_040000.pth \
-m /root/autodl-tmp/Inpainting/masks/irregular_mask/mask_rates_10_20 \
--save_frames -i /root/autodl-tmp/Inpainting/BundleFusion/B_fault/B_imgs_10_20/office1 \
--line ../geo_output/B_office1_10_20_02/line/ \
-o ../results/B_office1_10_20 --width 256 --height 256

python ../video_inference.py \
--ckpt_path ../experiments_model/propainter_train_LinePro_laplacianloss/gen_040000.pth \
-m /root/autodl-tmp/Inpainting/masks/irregular_mask/mask_rates_10_20 \
--save_frames -i /root/autodl-tmp/Inpainting/BundleFusion/B_fault/B_imgs_10_20/office2 \
--line ../geo_output/B_office2_10_20_02/line/ \
-o ../results/B_office2_10_20 --width 256 --height 256

python ../video_inference.py \
--ckpt_path ../experiments_model/propainter_train_LinePro_laplacianloss/gen_040000.pth \
-m /root/autodl-tmp/Inpainting/masks/irregular_mask/mask_rates_10_20 \
--save_frames -i /root/autodl-tmp/Inpainting/BundleFusion/B_fault/B_imgs_10_20/office3 \
--line ../geo_output/B_office3_10_20_02/line/ \
-o ../results/B_office3_10_20 --width 256 --height 256














##20_30
python ../video_inference.py \
--ckpt_path ../experiments_model/propainter_train_LinePro_laplacianloss/gen_040000.pth \
-m /root/autodl-tmp/Inpainting/masks/irregular_mask/mask_rates_20_30 \
--save_frames -i /root/autodl-tmp/Inpainting/BundleFusion/B_fault/B_imgs_20_30/office0 \
--line ../geo_output/B_office0_20_30_02/line/ \
-o ../results/B_office0_20_30 --width 256 --height 256

python ../video_inference.py \
--ckpt_path ../experiments_model/propainter_train_LinePro_laplacianloss/gen_040000.pth \
-m /root/autodl-tmp/Inpainting/masks/irregular_mask/mask_rates_20_30 \
--save_frames -i /root/autodl-tmp/Inpainting/BundleFusion/B_fault/B_imgs_20_30/office1 \
--line ../geo_output/B_office1_20_30_02/line/ \
-o ../results/B_office1_20_30 --width 256 --height 256

python ../video_inference.py \
--ckpt_path ../experiments_model/propainter_train_LinePro_laplacianloss/gen_040000.pth \
-m /root/autodl-tmp/Inpainting/masks/irregular_mask/mask_rates_20_30 \
--save_frames -i /root/autodl-tmp/Inpainting/BundleFusion/B_fault/B_imgs_20_30/office2 \
--line ../geo_output/B_office2_20_30_02/line/ \
-o ../results/B_office2_20_30 --width 256 --height 256

python ../video_inference.py \
--ckpt_path ../experiments_model/propainter_train_LinePro_laplacianloss/gen_040000.pth \
-m /root/autodl-tmp/Inpainting/masks/irregular_mask/mask_rates_20_30 \
--save_frames -i /root/autodl-tmp/Inpainting/BundleFusion/B_fault/B_imgs_20_30/office3 \
--line ../geo_output/B_office3_20_30_02/line/ \
-o ../results/B_office3_20_30 --width 256 --height 256















##30_40
python ../video_inference.py \
--ckpt_path ../experiments_model/propainter_train_LinePro_laplacianloss/gen_040000.pth \
-m /root/autodl-tmp/Inpainting/masks/irregular_mask/mask_rates_30_40 \
--save_frames -i /root/autodl-tmp/Inpainting/BundleFusion/B_fault/B_imgs_30_40/office0 \
--line ../geo_output/B_office0_30_40_02/line/ \
-o ../results/B_office0_30_40 --width 256 --height 256

python ../video_inference.py \
--ckpt_path ../experiments_model/propainter_train_LinePro_laplacianloss/gen_040000.pth \
-m /root/autodl-tmp/Inpainting/masks/irregular_mask/mask_rates_30_40 \
--save_frames -i /root/autodl-tmp/Inpainting/BundleFusion/B_fault/B_imgs_30_40/office1 \
--line ../geo_output/B_office1_30_40_02/line/ \
-o ../results/B_office1_30_40 --width 256 --height 256

python ../video_inference.py \
--ckpt_path ../experiments_model/propainter_train_LinePro_laplacianloss/gen_040000.pth \
-m /root/autodl-tmp/Inpainting/masks/irregular_mask/mask_rates_30_40 \
--save_frames -i /root/autodl-tmp/Inpainting/BundleFusion/B_fault/B_imgs_30_40/office2 \
--line ../geo_output/B_office2_30_40_02/line/ \
-o ../results/B_office2_30_40 --width 256 --height 256

python ../video_inference.py \
--ckpt_path ../experiments_model/propainter_train_LinePro_laplacianloss/gen_040000.pth \
-m /root/autodl-tmp/Inpainting/masks/irregular_mask/mask_rates_30_40 \
--save_frames -i /root/autodl-tmp/Inpainting/BundleFusion/B_fault/B_imgs_30_40/office3 \
--line ../geo_output/B_office3_30_40_02/line/ \
-o ../results/B_office3_30_40 --width 256 --height 256















##40_50
python ../video_inference.py \
--ckpt_path ../experiments_model/propainter_train_LinePro_laplacianloss/gen_040000.pth \
-m /root/autodl-tmp/Inpainting/masks/irregular_mask/mask_rates_40_50 \
--save_frames -i /root/autodl-tmp/Inpainting/BundleFusion/B_fault/B_imgs_40_50/office0 \
--line ../geo_output/B_office0_40_50_02/line/ \
-o ../results/B_office0_40_50 --width 256 --height 256

python ../video_inference.py \
--ckpt_path ../experiments_model/propainter_train_LinePro_laplacianloss/gen_040000.pth \
-m /root/autodl-tmp/Inpainting/masks/irregular_mask/mask_rates_40_50 \
--save_frames -i /root/autodl-tmp/Inpainting/BundleFusion/B_fault/B_imgs_40_50/office1 \
--line ../geo_output/B_office1_40_50_02/line/ \
-o ../results/B_office1_40_50 --width 256 --height 256

python ../video_inference.py \
--ckpt_path ../experiments_model/propainter_train_LinePro_laplacianloss/gen_040000.pth \
-m /root/autodl-tmp/Inpainting/masks/irregular_mask/mask_rates_40_50 \
--save_frames -i /root/autodl-tmp/Inpainting/BundleFusion/B_fault/B_imgs_40_50/office2 \
--line ../geo_output/B_office2_40_50_02/line/ \
-o ../results/B_office2_40_50 --width 256 --height 256

python ../video_inference.py \
--ckpt_path ../experiments_model/propainter_train_LinePro_laplacianloss/gen_040000.pth \
-m /root/autodl-tmp/Inpainting/masks/irregular_mask/mask_rates_40_50 \
--save_frames -i /root/autodl-tmp/Inpainting/BundleFusion/B_fault/B_imgs_40_50/office3 \
--line ../geo_output/B_office3_40_50_02/line/ \
-o ../results/B_office3_40_50 --width 256 --height 256
