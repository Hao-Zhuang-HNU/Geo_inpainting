##5-10
# python eval_video_inpainting_metrics.py --pre_path ./results/Dbest_f5d7c3_mask_5_10 \
# --gt_path  ../DL3DV/D_val/D_imgs/f5d7c37dd2779b1b88782a8f1074c21b8d5f635a6a1b8fc3f6a2ae17b7bbb601/images_8 \
# --mask_path ../masks/irregular_mask/mask_rates_5_10 \
# --pix 256 --resize --out ./eval_out_video/Dbest_f5d7c3_mask_5_10

# python eval_video_inpainting_metrics.py --pre_path ./results/Dbest_f56bbd_mask_5_10 \
# --gt_path  ../DL3DV/D_val/D_imgs/f56bbd8d4b1b756618432b94d15f5149ec94a2d808d7d03bcf335031d2149e2c/images_8 \
# --mask_path ../masks/irregular_mask/mask_rates_5_10 \
# --pix 256 --resize --out ./eval_out_video/Dbest_f56bbd_mask_5_10

# python eval_video_inpainting_metrics.py --pre_path ./results/Dbest_f080ed_mask_5_10 \
# --gt_path  ../DL3DV/D_val/D_imgs/f080ed61285ea79eec0a1c52341e0e8042ea869fed0582cb2d17b6d8b26501e4/images_8 \
# --mask_path ../masks/irregular_mask/mask_rates_5_10 \
# --pix 256 --resize --out ./eval_out_video/Dbest_f080ed_mask_5_10

# python eval_video_inpainting_metrics.py --pre_path ./results/Dbest_f672ea_mask_5_10 \
# --gt_path  ../DL3DV/D_val/D_imgs/f672ea1d22049976baa370bb881795d51fe2155b28fc9f63325fcf61b8f218cd/images_8 \
# --mask_path ../masks/irregular_mask/mask_rates_5_10 \
# --pix 256 --resize --out ./eval_out_video/Dbest_f672ea_mask_5_10

# python eval_video_inpainting_metrics.py --pre_path ./results/Dbest_f49418_mask_5_10 \
# --gt_path  ../DL3DV/D_val/D_imgs/f49418591f84e7a24711c67aa05b3074010af65debb06042adb444d806af9965/images_8 \
# --mask_path ../masks/irregular_mask/mask_rates_5_10 \
# --pix 256 --resize --out ./eval_out_video/Dbest_f49418_mask_5_10

# python eval_video_inpainting_metrics.py --pre_path ./results/Dbest_f81358_mask_5_10 \
# --gt_path  ../DL3DV/D_val/D_imgs/f8135871bc4577d7a81857c0684e354444e5c8927b8efcbf2b0bdd22dbfb9ce7/images_8 \
# --mask_path ../masks/irregular_mask/mask_rates_5_10 \
# --pix 256 --resize --out ./eval_out_video/Dbest_f81358_mask_5_10

# python eval_video_inpainting_metrics.py --pre_path ./results/Dbest_fac184_mask_5_10 \
# --gt_path  ../DL3DV/D_val/D_imgs/fac184210decc5fb0df8afd4962543bbce7f6e461ee413c3c949d2e783b4fa21/images_8 \
# --mask_path ../masks/irregular_mask/mask_rates_5_10 \
# --pix 256 --resize --out ./eval_out_video/Dbest_fac184_mask_5_10

# python eval_video_inpainting_metrics.py --pre_path ./results/Dbest_fbb7b6_mask_5_10 \
# --gt_path  ../DL3DV/D_val/D_imgs/fbb7b6b3a9ee0c8f86f6a9af413c2c628155403b2b566df6510af01a1784fa41/images_8 \
# --mask_path ../masks/irregular_mask/mask_rates_5_10 \
# --pix 256 --resize --out ./eval_out_video/Dbest_fbb7b6_mask_5_10

# python eval_video_inpainting_metrics.py --pre_path ./results/Dbest_fc18ec_mask_5_10 \
# --gt_path  ../DL3DV/D_val/D_imgs/fc18ec3166486f91bf72696d9f9ab72adb2bc1cde8f5b773766e0e7564880e46/images_8 \
# --mask_path ../masks/irregular_mask/mask_rates_5_10 \
# --pix 256 --resize --out ./eval_out_video/Dbest_fc18ec_mask_5_10

# python eval_video_inpainting_metrics.py --pre_path ./results/Dbest_fe30a8_mask_5_10 \
# --gt_path  ../DL3DV/D_val/D_imgs/fe30a8657df5caa5e33f6c128dc3bc029d337395e9f3d7a17d1ce85ebe83a5c5/images_8 \
# --mask_path ../masks/irregular_mask/mask_rates_5_10 \
# --pix 256 --resize --out ./eval_out_video/Dbest_fe30a8_mask_5_10

python eval_video_inpainting_metrics_shift.py --pre_path ./results/Dbest_f5d7c3_mask_5_10 \
--gt_path  ../DL3DV/D_val/D_imgs/f5d7c37dd2779b1b88782a8f1074c21b8d5f635a6a1b8fc3f6a2ae17b7bbb601/images_8 \
--mask_path ../masks/irregular_mask/mask_rates_5_10 \
--debug \
--out ./eval_out_video/Dbest_f5d7c3_mask_5_10 \
--mask_shift 2

python eval_video_inpainting_metrics_shift.py --pre_path ./results/Dbest_f56bbd_mask_5_10 \
--gt_path  ../DL3DV/D_val/D_imgs/f56bbd8d4b1b756618432b94d15f5149ec94a2d808d7d03bcf335031d2149e2c/images_8 \
--mask_path ../masks/irregular_mask/mask_rates_5_10 \
--pix 256 --debug --mask_shift 1 --out ./eval_out_video/Dbest_f56bbd_mask_5_10

python eval_video_inpainting_metrics_shift.py --pre_path ./results/Dbest_f080ed_mask_5_10 \
--gt_path  ../DL3DV/D_val/D_imgs/f080ed61285ea79eec0a1c52341e0e8042ea869fed0582cb2d17b6d8b26501e4/images_8 \
--mask_path ../masks/irregular_mask/mask_rates_5_10 \
--pix 256 --debug --mask_shift 1 --out ./eval_out_video/Dbest_f080ed_mask_5_10

python eval_video_inpainting_metrics_shift.py --pre_path ./results/Dbest_f672ea_mask_5_10 \
--gt_path  ../DL3DV/D_val/D_imgs/f672ea1d22049976baa370bb881795d51fe2155b28fc9f63325fcf61b8f218cd/images_8 \
--mask_path ../masks/irregular_mask/mask_rates_5_10 \
--pix 256 --debug --mask_shift 1 --out ./eval_out_video/Dbest_f672ea_mask_5_10

python eval_video_inpainting_metrics_shift.py --pre_path ./results/Dbest_f49418_mask_5_10 \
--gt_path  ../DL3DV/D_val/D_imgs/f49418591f84e7a24711c67aa05b3074010af65debb06042adb444d806af9965/images_8 \
--mask_path ../masks/irregular_mask/mask_rates_5_10 \
--pix 256 --debug --mask_shift 1 --out ./eval_out_video/Dbest_f49418_mask_5_10

python eval_video_inpainting_metrics_shift.py --pre_path ./results/Dbest_f81358_mask_5_10 \
--gt_path  ../DL3DV/D_val/D_imgs/f8135871bc4577d7a81857c0684e354444e5c8927b8efcbf2b0bdd22dbfb9ce7/images_8 \
--mask_path ../masks/irregular_mask/mask_rates_5_10 \
--pix 256 --debug --mask_shift 1 --out ./eval_out_video/Dbest_f81358_mask_5_10

python eval_video_inpainting_metrics_shift.py --pre_path ./results/Dbest_fac184_mask_5_10 \
--gt_path  ../DL3DV/D_val/D_imgs/fac184210decc5fb0df8afd4962543bbce7f6e461ee413c3c949d2e783b4fa21/images_8 \
--mask_path ../masks/irregular_mask/mask_rates_5_10 \
--pix 256 --debug --mask_shift 1 --out ./eval_out_video/Dbest_fac184_mask_5_10

python eval_video_inpainting_metrics_shift.py --pre_path ./results/Dbest_fbb7b6_mask_5_10 \
--gt_path  ../DL3DV/D_val/D_imgs/fbb7b6b3a9ee0c8f86f6a9af413c2c628155403b2b566df6510af01a1784fa41/images_8 \
--mask_path ../masks/irregular_mask/mask_rates_5_10 \
--pix 256 --debug --mask_shift 1 --out ./eval_out_video/Dbest_fbb7b6_mask_5_10

python eval_video_inpainting_metrics_shift.py --pre_path ./results/Dbest_fc18ec_mask_5_10 \
--gt_path  ../DL3DV/D_val/D_imgs/fc18ec3166486f91bf72696d9f9ab72adb2bc1cde8f5b773766e0e7564880e46/images_8 \
--mask_path ../masks/irregular_mask/mask_rates_5_10 \
--pix 256 --debug --mask_shift 1 --out ./eval_out_video/Dbest_fc18ec_mask_5_10

python eval_video_inpainting_metrics_shift.py --pre_path ./results/Dbest_fe30a8_mask_5_10 \
--gt_path  ../DL3DV/D_val/D_imgs/fe30a8657df5caa5e33f6c128dc3bc029d337395e9f3d7a17d1ce85ebe83a5c5/images_8 \
--mask_path ../masks/irregular_mask/mask_rates_5_10 \
--pix 256 --debug --mask_shift 1 --out ./eval_out_video/Dbest_fe30a8_mask_5_10