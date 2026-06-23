##mask_5_10
# python ../line_inference.py \
#   --GPU_ids 0 \
#   --ckpt_path ../ckpt/10cur_latest.pth \
#   --hawp_ckpt_path ../ckpt/best_lsm_hawp.pth \
#   --imgs_list ../ScannetPP_longlist/ScannetPP_longlist_f1e01a_imgs.txt \
#   --mask_list ../data_list/irregular_mask_5_10.txt \
#   --save_url ../geo_output/Slong_mask_5_10/f1e01a \
# --solid_line --line_binary_thresh 0.2 \
#   --amp_inference \
#   --data_num_workers 8 \
#   --skip_save_edge
# python ../line_inference.py \
#   --GPU_ids 0 \
#   --ckpt_path ../ckpt/10cur_latest.pth \
#   --hawp_ckpt_path ../ckpt/best_lsm_hawp.pth \
#   --imgs_list ../ScannetPP_longlist/ScannetPP_longlist_f34d53_imgs.txt \
#   --mask_list ../data_list/irregular_mask_5_10.txt \
#   --save_url ../geo_output/Slong_mask_5_10/f34d53 \
# --solid_line --line_binary_thresh 0.2 \
#   --amp_inference \
#   --data_num_workers 8 \
#   --skip_save_edge
# python ../line_inference.py \
#   --GPU_ids 0 \
#   --ckpt_path ../ckpt/10cur_latest.pth \
#   --hawp_ckpt_path ../ckpt/best_lsm_hawp.pth \
#   --imgs_list ../ScannetPP_longlist/ScannetPP_longlist_f36e3e_imgs.txt \
#   --mask_list ../data_list/irregular_mask_5_10.txt \
#   --save_url ../geo_output/Slong_mask_5_10/f36e3e \
# --solid_line --line_binary_thresh 0.2 \
#   --amp_inference \
#   --data_num_workers 8 \
#   --skip_save_edge
# python ../line_inference.py \
#   --GPU_ids 0 \
#   --ckpt_path ../ckpt/10cur_latest.pth \
#   --hawp_ckpt_path ../ckpt/best_lsm_hawp.pth \
#   --imgs_list ../ScannetPP_longlist/ScannetPP_longlist_f38b01_imgs.txt \
#   --mask_list ../data_list/irregular_mask_5_10.txt \
#   --save_url ../geo_output/Slong_mask_5_10/f38b01 \
# --solid_line --line_binary_thresh 0.2 \
#   --amp_inference \
#   --data_num_workers 8 \
#   --skip_save_edge
# python ../line_inference.py \
#   --GPU_ids 0 \
#   --ckpt_path ../ckpt/10cur_latest.pth \
#   --hawp_ckpt_path ../ckpt/best_lsm_hawp.pth \
#   --imgs_list ../ScannetPP_longlist/ScannetPP_longlist_f5726e_imgs.txt \
#   --mask_list ../data_list/irregular_mask_5_10.txt \
#   --save_url ../geo_output/Slong_mask_5_10/f5726e \
# --solid_line --line_binary_thresh 0.2 \
#   --amp_inference \
#   --data_num_workers 8 \
#   --skip_save_edge
# python ../line_inference.py \
#   --GPU_ids 0 \
#   --ckpt_path ../ckpt/10cur_latest.pth \
#   --hawp_ckpt_path ../ckpt/best_lsm_hawp.pth \
#   --imgs_list ../ScannetPP_longlist/ScannetPP_longlist_f84708_imgs.txt \
#   --mask_list ../data_list/irregular_mask_5_10.txt \
#   --save_url ../geo_output/Slong_mask_5_10/f84708 \
# --solid_line --line_binary_thresh 0.2 \
#   --amp_inference \
#   --data_num_workers 8 \
#   --skip_save_edge
# python ../line_inference.py \
#   --GPU_ids 0 \
#   --ckpt_path ../ckpt/10cur_latest.pth \
#   --hawp_ckpt_path ../ckpt/best_lsm_hawp.pth \
#   --imgs_list ../ScannetPP_longlist/ScannetPP_longlist_f9397a_imgs.txt \
#   --mask_list ../data_list/irregular_mask_5_10.txt \
#   --save_url ../geo_output/Slong_mask_5_10/f9397a \
# --solid_line --line_binary_thresh 0.2 \
#   --amp_inference \
#   --data_num_workers 8 \
#   --skip_save_edge
# python ../line_inference.py \
#   --GPU_ids 0 \
#   --ckpt_path ../ckpt/10cur_latest.pth \
#   --hawp_ckpt_path ../ckpt/best_lsm_hawp.pth \
#   --imgs_list ../ScannetPP_longlist/ScannetPP_longlist_faba6e_imgs.txt \
#   --mask_list ../data_list/irregular_mask_5_10.txt \
#   --save_url ../geo_output/Slong_mask_5_10/faba6e \
# --solid_line --line_binary_thresh 0.2 \
#   --amp_inference \
#   --data_num_workers 8 \
#   --skip_save_edge











##mask_10_20
python ../line_inference.py \
  --GPU_ids 0 \
  --ckpt_path ../ckpt/10cur_latest.pth \
  --hawp_ckpt_path ../ckpt/best_lsm_hawp.pth \
  --imgs_list ../ScannetPP_longlist/ScannetPP_longlist_f1e01a_imgs.txt \
  --mask_list ../data_list/irregular_mask_10_20.txt \
  --save_url ../geo_output/Slong_mask_10_20/f1e01a \
  --test_line_list ../ScannetPP_longlist/ScannetPP_longlist_f1e01a_pkls.txt \
--solid_line --line_binary_thresh 0.2 \
  --amp_inference \
  --data_num_workers 8 \
  --skip_save_edge
python ../line_inference.py \
  --GPU_ids 0 \
  --ckpt_path ../ckpt/10cur_latest.pth \
  --hawp_ckpt_path ../ckpt/best_lsm_hawp.pth \
  --imgs_list ../ScannetPP_longlist/ScannetPP_longlist_f34d53_imgs.txt \
  --mask_list ../data_list/irregular_mask_10_20.txt \
  --save_url ../geo_output/Slong_mask_10_20/f34d53 \
  --test_line_list ../ScannetPP_longlist/ScannetPP_longlist_f34d53_pkls.txt \
--solid_line --line_binary_thresh 0.2 \
  --amp_inference \
  --data_num_workers 8 \
  --skip_save_edge
python ../line_inference.py \
  --GPU_ids 0 \
  --ckpt_path ../ckpt/10cur_latest.pth \
  --hawp_ckpt_path ../ckpt/best_lsm_hawp.pth \
  --imgs_list ../ScannetPP_longlist/ScannetPP_longlist_f36e3e_imgs.txt \
  --mask_list ../data_list/irregular_mask_10_20.txt \
  --save_url ../geo_output/Slong_mask_10_20/f36e3e \
--solid_line --line_binary_thresh 0.2 \
--test_line_list ../ScannetPP_longlist/ScannetPP_longlist_f36e3e_pkls.txt \
  --amp_inference \
  --data_num_workers 8 \
  --skip_save_edge
python ../line_inference.py \
  --GPU_ids 0 \
  --ckpt_path ../ckpt/10cur_latest.pth \
  --hawp_ckpt_path ../ckpt/best_lsm_hawp.pth \
  --imgs_list ../ScannetPP_longlist/ScannetPP_longlist_f38b01_imgs.txt \
  --test_line_list ../ScannetPP_longlist/ScannetPP_longlist_f38b01_pkls.txt \
  --mask_list ../data_list/irregular_mask_10_20.txt \
  --save_url ../geo_output/Slong_mask_10_20/f38b01 \
--solid_line --line_binary_thresh 0.2 \
  --amp_inference \
  --data_num_workers 8 \
  --skip_save_edge
python ../line_inference.py \
  --GPU_ids 0 \
  --ckpt_path ../ckpt/10cur_latest.pth \
  --hawp_ckpt_path ../ckpt/best_lsm_hawp.pth \
  --imgs_list ../ScannetPP_longlist/ScannetPP_longlist_f5726e_imgs.txt \
  --test_line_list ../ScannetPP_longlist/ScannetPP_longlist_f5726e_pkls.txt \
  --mask_list ../data_list/irregular_mask_10_20.txt \
  --save_url ../geo_output/Slong_mask_10_20/f5726e \
--solid_line --line_binary_thresh 0.2 \
  --amp_inference \
  --data_num_workers 8 \
  --skip_save_edge
python ../line_inference.py \
  --GPU_ids 0 \
  --ckpt_path ../ckpt/10cur_latest.pth \
  --hawp_ckpt_path ../ckpt/best_lsm_hawp.pth \
  --imgs_list ../ScannetPP_longlist/ScannetPP_longlist_f84708_imgs.txt \
  --test_line_list ../ScannetPP_longlist/ScannetPP_longlist_f84708_pkls.txt \
  --mask_list ../data_list/irregular_mask_10_20.txt \
  --save_url ../geo_output/Slong_mask_10_20/f84708 \
--solid_line --line_binary_thresh 0.2 \
  --amp_inference \
  --data_num_workers 8 \
  --skip_save_edge
python ../line_inference.py \
  --GPU_ids 0 \
  --ckpt_path ../ckpt/10cur_latest.pth \
  --hawp_ckpt_path ../ckpt/best_lsm_hawp.pth \
  --imgs_list ../ScannetPP_longlist/ScannetPP_longlist_f9397a_imgs.txt \
  --test_line_list ../ScannetPP_longlist/ScannetPP_longlist_f9397a_pkls.txt \
  --mask_list ../data_list/irregular_mask_10_20.txt \
  --save_url ../geo_output/Slong_mask_10_20/f9397a \
--solid_line --line_binary_thresh 0.2 \
  --amp_inference \
  --data_num_workers 8 \
  --skip_save_edge
python ../line_inference.py \
  --GPU_ids 0 \
  --ckpt_path ../ckpt/10cur_latest.pth \
  --hawp_ckpt_path ../ckpt/best_lsm_hawp.pth \
  --imgs_list ../ScannetPP_longlist/ScannetPP_longlist_faba6e_imgs.txt \
  --test_line_list ../ScannetPP_longlist/ScannetPP_longlist_faba6e_pkls.txt \
  --mask_list ../data_list/irregular_mask_10_20.txt \
  --save_url ../geo_output/Slong_mask_10_20/faba6e \
--solid_line --line_binary_thresh 0.2 \
  --amp_inference \
  --data_num_workers 8 \
  --skip_save_edge








##mask_20_30
python ../line_inference.py \
  --GPU_ids 0 \
  --ckpt_path ../ckpt/10cur_latest.pth \
  --hawp_ckpt_path ../ckpt/best_lsm_hawp.pth \
  --imgs_list ../ScannetPP_longlist/ScannetPP_longlist_f1e01a_imgs.txt \
  --mask_list ../data_list/irregular_mask_20_30.txt \
  --save_url ../geo_output/Slong_mask_20_30/f1e01a \
  --test_line_list ../ScannetPP_longlist/ScannetPP_longlist_f1e01a_pkls.txt \
--solid_line --line_binary_thresh 0.2 \
  --amp_inference \
  --data_num_workers 8 \
  --skip_save_edge
python ../line_inference.py \
  --GPU_ids 0 \
  --ckpt_path ../ckpt/10cur_latest.pth \
  --hawp_ckpt_path ../ckpt/best_lsm_hawp.pth \
  --imgs_list ../ScannetPP_longlist/ScannetPP_longlist_f34d53_imgs.txt \
  --mask_list ../data_list/irregular_mask_20_30.txt \
  --save_url ../geo_output/Slong_mask_20_30/f34d53 \
  --test_line_list ../ScannetPP_longlist/ScannetPP_longlist_f34d53_pkls.txt \
--solid_line --line_binary_thresh 0.2 \
  --amp_inference \
  --data_num_workers 8 \
  --skip_save_edge
python ../line_inference.py \
  --GPU_ids 0 \
  --ckpt_path ../ckpt/10cur_latest.pth \
  --hawp_ckpt_path ../ckpt/best_lsm_hawp.pth \
  --imgs_list ../ScannetPP_longlist/ScannetPP_longlist_f36e3e_imgs.txt \
  --mask_list ../data_list/irregular_mask_20_30.txt \
  --save_url ../geo_output/Slong_mask_20_30/f36e3e \
--solid_line --line_binary_thresh 0.2 \
--test_line_list ../ScannetPP_longlist/ScannetPP_longlist_f36e3e_pkls.txt \
  --amp_inference \
  --data_num_workers 8 \
  --skip_save_edge
python ../line_inference.py \
  --GPU_ids 0 \
  --ckpt_path ../ckpt/10cur_latest.pth \
  --hawp_ckpt_path ../ckpt/best_lsm_hawp.pth \
  --imgs_list ../ScannetPP_longlist/ScannetPP_longlist_f38b01_imgs.txt \
  --test_line_list ../ScannetPP_longlist/ScannetPP_longlist_f38b01_pkls.txt \
  --mask_list ../data_list/irregular_mask_20_30.txt \
  --save_url ../geo_output/Slong_mask_20_30/f38b01 \
--solid_line --line_binary_thresh 0.2 \
  --amp_inference \
  --data_num_workers 8 \
  --skip_save_edge
python ../line_inference.py \
  --GPU_ids 0 \
  --ckpt_path ../ckpt/10cur_latest.pth \
  --hawp_ckpt_path ../ckpt/best_lsm_hawp.pth \
  --imgs_list ../ScannetPP_longlist/ScannetPP_longlist_f5726e_imgs.txt \
  --test_line_list ../ScannetPP_longlist/ScannetPP_longlist_f5726e_pkls.txt \
  --mask_list ../data_list/irregular_mask_20_30.txt \
  --save_url ../geo_output/Slong_mask_20_30/f5726e \
--solid_line --line_binary_thresh 0.2 \
  --amp_inference \
  --data_num_workers 8 \
  --skip_save_edge
python ../line_inference.py \
  --GPU_ids 0 \
  --ckpt_path ../ckpt/10cur_latest.pth \
  --hawp_ckpt_path ../ckpt/best_lsm_hawp.pth \
  --imgs_list ../ScannetPP_longlist/ScannetPP_longlist_f84708_imgs.txt \
  --test_line_list ../ScannetPP_longlist/ScannetPP_longlist_f84708_pkls.txt \
  --mask_list ../data_list/irregular_mask_20_30.txt \
  --save_url ../geo_output/Slong_mask_20_30/f84708 \
--solid_line --line_binary_thresh 0.2 \
  --amp_inference \
  --data_num_workers 8 \
  --skip_save_edge
python ../line_inference.py \
  --GPU_ids 0 \
  --ckpt_path ../ckpt/10cur_latest.pth \
  --hawp_ckpt_path ../ckpt/best_lsm_hawp.pth \
  --imgs_list ../ScannetPP_longlist/ScannetPP_longlist_f9397a_imgs.txt \
  --test_line_list ../ScannetPP_longlist/ScannetPP_longlist_f9397a_pkls.txt \
  --mask_list ../data_list/irregular_mask_20_30.txt \
  --save_url ../geo_output/Slong_mask_20_30/f9397a \
--solid_line --line_binary_thresh 0.2 \
  --amp_inference \
  --data_num_workers 8 \
  --skip_save_edge
python ../line_inference.py \
  --GPU_ids 0 \
  --ckpt_path ../ckpt/10cur_latest.pth \
  --hawp_ckpt_path ../ckpt/best_lsm_hawp.pth \
  --imgs_list ../ScannetPP_longlist/ScannetPP_longlist_faba6e_imgs.txt \
  --test_line_list ../ScannetPP_longlist/ScannetPP_longlist_faba6e_pkls.txt \
  --mask_list ../data_list/irregular_mask_20_30.txt \
  --save_url ../geo_output/Slong_mask_20_30/faba6e \
--solid_line --line_binary_thresh 0.2 \
  --amp_inference \
  --data_num_workers 8 \
  --skip_save_edge



















##mask_30_40
python ../line_inference.py \
  --GPU_ids 0 \
  --ckpt_path ../ckpt/10cur_latest.pth \
  --hawp_ckpt_path ../ckpt/best_lsm_hawp.pth \
  --imgs_list ../ScannetPP_longlist/ScannetPP_longlist_f1e01a_imgs.txt \
  --mask_list ../data_list/irregular_mask_30_40.txt \
  --save_url ../geo_output/Slong_mask_30_40/f1e01a \
  --test_line_list ../ScannetPP_longlist/ScannetPP_longlist_f1e01a_pkls.txt \
--solid_line --line_binary_thresh 0.2 \
  --amp_inference \
  --data_num_workers 8 \
  --skip_save_edge
python ../line_inference.py \
  --GPU_ids 0 \
  --ckpt_path ../ckpt/10cur_latest.pth \
  --hawp_ckpt_path ../ckpt/best_lsm_hawp.pth \
  --imgs_list ../ScannetPP_longlist/ScannetPP_longlist_f34d53_imgs.txt \
  --mask_list ../data_list/irregular_mask_30_40.txt \
  --save_url ../geo_output/Slong_mask_30_40/f34d53 \
  --test_line_list ../ScannetPP_longlist/ScannetPP_longlist_f34d53_pkls.txt \
--solid_line --line_binary_thresh 0.2 \
  --amp_inference \
  --data_num_workers 8 \
  --skip_save_edge
python ../line_inference.py \
  --GPU_ids 0 \
  --ckpt_path ../ckpt/10cur_latest.pth \
  --hawp_ckpt_path ../ckpt/best_lsm_hawp.pth \
  --imgs_list ../ScannetPP_longlist/ScannetPP_longlist_f36e3e_imgs.txt \
  --mask_list ../data_list/irregular_mask_30_40.txt \
  --save_url ../geo_output/Slong_mask_30_40/f36e3e \
--solid_line --line_binary_thresh 0.2 \
--test_line_list ../ScannetPP_longlist/ScannetPP_longlist_f36e3e_pkls.txt \
  --amp_inference \
  --data_num_workers 8 \
  --skip_save_edge
python ../line_inference.py \
  --GPU_ids 0 \
  --ckpt_path ../ckpt/10cur_latest.pth \
  --hawp_ckpt_path ../ckpt/best_lsm_hawp.pth \
  --imgs_list ../ScannetPP_longlist/ScannetPP_longlist_f38b01_imgs.txt \
  --test_line_list ../ScannetPP_longlist/ScannetPP_longlist_f38b01_pkls.txt \
  --mask_list ../data_list/irregular_mask_30_40.txt \
  --save_url ../geo_output/Slong_mask_30_40/f38b01 \
--solid_line --line_binary_thresh 0.2 \
  --amp_inference \
  --data_num_workers 8 \
  --skip_save_edge
python ../line_inference.py \
  --GPU_ids 0 \
  --ckpt_path ../ckpt/10cur_latest.pth \
  --hawp_ckpt_path ../ckpt/best_lsm_hawp.pth \
  --imgs_list ../ScannetPP_longlist/ScannetPP_longlist_f5726e_imgs.txt \
  --test_line_list ../ScannetPP_longlist/ScannetPP_longlist_f5726e_pkls.txt \
  --mask_list ../data_list/irregular_mask_30_40.txt \
  --save_url ../geo_output/Slong_mask_30_40/f5726e \
--solid_line --line_binary_thresh 0.2 \
  --amp_inference \
  --data_num_workers 8 \
  --skip_save_edge
python ../line_inference.py \
  --GPU_ids 0 \
  --ckpt_path ../ckpt/10cur_latest.pth \
  --hawp_ckpt_path ../ckpt/best_lsm_hawp.pth \
  --imgs_list ../ScannetPP_longlist/ScannetPP_longlist_f84708_imgs.txt \
  --test_line_list ../ScannetPP_longlist/ScannetPP_longlist_f84708_pkls.txt \
  --mask_list ../data_list/irregular_mask_30_40.txt \
  --save_url ../geo_output/Slong_mask_30_40/f84708 \
--solid_line --line_binary_thresh 0.2 \
  --amp_inference \
  --data_num_workers 8 \
  --skip_save_edge
python ../line_inference.py \
  --GPU_ids 0 \
  --ckpt_path ../ckpt/10cur_latest.pth \
  --hawp_ckpt_path ../ckpt/best_lsm_hawp.pth \
  --imgs_list ../ScannetPP_longlist/ScannetPP_longlist_f9397a_imgs.txt \
  --test_line_list ../ScannetPP_longlist/ScannetPP_longlist_f9397a_pkls.txt \
  --mask_list ../data_list/irregular_mask_30_40.txt \
  --save_url ../geo_output/Slong_mask_30_40/f9397a \
--solid_line --line_binary_thresh 0.2 \
  --amp_inference \
  --data_num_workers 8 \
  --skip_save_edge
python ../line_inference.py \
  --GPU_ids 0 \
  --ckpt_path ../ckpt/10cur_latest.pth \
  --hawp_ckpt_path ../ckpt/best_lsm_hawp.pth \
  --imgs_list ../ScannetPP_longlist/ScannetPP_longlist_faba6e_imgs.txt \
  --test_line_list ../ScannetPP_longlist/ScannetPP_longlist_faba6e_pkls.txt \
  --mask_list ../data_list/irregular_mask_30_40.txt \
  --save_url ../geo_output/Slong_mask_30_40/faba6e \
--solid_line --line_binary_thresh 0.2 \
  --amp_inference \
  --data_num_workers 8 \
  --skip_save_edge













##mask_40_50
python ../line_inference.py \
  --GPU_ids 0 \
  --ckpt_path ../ckpt/10cur_latest.pth \
  --hawp_ckpt_path ../ckpt/best_lsm_hawp.pth \
  --imgs_list ../ScannetPP_longlist/ScannetPP_longlist_f1e01a_imgs.txt \
  --mask_list ../data_list/irregular_mask_40_50.txt \
  --save_url ../geo_output/Slong_mask_40_50/f1e01a \
  --test_line_list ../ScannetPP_longlist/ScannetPP_longlist_f1e01a_pkls.txt \
--solid_line --line_binary_thresh 0.2 \
  --amp_inference \
  --data_num_workers 8 \
  --skip_save_edge
python ../line_inference.py \
  --GPU_ids 0 \
  --ckpt_path ../ckpt/10cur_latest.pth \
  --hawp_ckpt_path ../ckpt/best_lsm_hawp.pth \
  --imgs_list ../ScannetPP_longlist/ScannetPP_longlist_f34d53_imgs.txt \
  --mask_list ../data_list/irregular_mask_40_50.txt \
  --save_url ../geo_output/Slong_mask_40_50/f34d53 \
  --test_line_list ../ScannetPP_longlist/ScannetPP_longlist_f34d53_pkls.txt \
--solid_line --line_binary_thresh 0.2 \
  --amp_inference \
  --data_num_workers 8 \
  --skip_save_edge
python ../line_inference.py \
  --GPU_ids 0 \
  --ckpt_path ../ckpt/10cur_latest.pth \
  --hawp_ckpt_path ../ckpt/best_lsm_hawp.pth \
  --imgs_list ../ScannetPP_longlist/ScannetPP_longlist_f36e3e_imgs.txt \
  --mask_list ../data_list/irregular_mask_40_50.txt \
  --save_url ../geo_output/Slong_mask_40_50/f36e3e \
--solid_line --line_binary_thresh 0.2 \
--test_line_list ../ScannetPP_longlist/ScannetPP_longlist_f36e3e_pkls.txt \
  --amp_inference \
  --data_num_workers 8 \
  --skip_save_edge
python ../line_inference.py \
  --GPU_ids 0 \
  --ckpt_path ../ckpt/10cur_latest.pth \
  --hawp_ckpt_path ../ckpt/best_lsm_hawp.pth \
  --imgs_list ../ScannetPP_longlist/ScannetPP_longlist_f38b01_imgs.txt \
  --test_line_list ../ScannetPP_longlist/ScannetPP_longlist_f38b01_pkls.txt \
  --mask_list ../data_list/irregular_mask_40_50.txt \
  --save_url ../geo_output/Slong_mask_40_50/f38b01 \
--solid_line --line_binary_thresh 0.2 \
  --amp_inference \
  --data_num_workers 8 \
  --skip_save_edge
python ../line_inference.py \
  --GPU_ids 0 \
  --ckpt_path ../ckpt/10cur_latest.pth \
  --hawp_ckpt_path ../ckpt/best_lsm_hawp.pth \
  --imgs_list ../ScannetPP_longlist/ScannetPP_longlist_f5726e_imgs.txt \
  --test_line_list ../ScannetPP_longlist/ScannetPP_longlist_f5726e_pkls.txt \
  --mask_list ../data_list/irregular_mask_40_50.txt \
  --save_url ../geo_output/Slong_mask_40_50/f5726e \
--solid_line --line_binary_thresh 0.2 \
  --amp_inference \
  --data_num_workers 8 \
  --skip_save_edge
python ../line_inference.py \
  --GPU_ids 0 \
  --ckpt_path ../ckpt/10cur_latest.pth \
  --hawp_ckpt_path ../ckpt/best_lsm_hawp.pth \
  --imgs_list ../ScannetPP_longlist/ScannetPP_longlist_f84708_imgs.txt \
  --test_line_list ../ScannetPP_longlist/ScannetPP_longlist_f84708_pkls.txt \
  --mask_list ../data_list/irregular_mask_40_50.txt \
  --save_url ../geo_output/Slong_mask_40_50/f84708 \
--solid_line --line_binary_thresh 0.2 \
  --amp_inference \
  --data_num_workers 8 \
  --skip_save_edge
python ../line_inference.py \
  --GPU_ids 0 \
  --ckpt_path ../ckpt/10cur_latest.pth \
  --hawp_ckpt_path ../ckpt/best_lsm_hawp.pth \
  --imgs_list ../ScannetPP_longlist/ScannetPP_longlist_f9397a_imgs.txt \
  --test_line_list ../ScannetPP_longlist/ScannetPP_longlist_f9397a_pkls.txt \
  --mask_list ../data_list/irregular_mask_40_50.txt \
  --save_url ../geo_output/Slong_mask_40_50/f9397a \
--solid_line --line_binary_thresh 0.2 \
  --amp_inference \
  --data_num_workers 8 \
  --skip_save_edge
python ../line_inference.py \
  --GPU_ids 0 \
  --ckpt_path ../ckpt/10cur_latest.pth \
  --hawp_ckpt_path ../ckpt/best_lsm_hawp.pth \
  --imgs_list ../ScannetPP_longlist/ScannetPP_longlist_faba6e_imgs.txt \
  --test_line_list ../ScannetPP_longlist/ScannetPP_longlist_faba6e_pkls.txt \
  --mask_list ../data_list/irregular_mask_40_50.txt \
  --save_url ../geo_output/Slong_mask_40_50/faba6e \
--solid_line --line_binary_thresh 0.2 \
  --amp_inference \
  --data_num_workers 8 \
  --skip_save_edge










