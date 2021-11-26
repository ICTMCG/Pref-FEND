# ========================= Pattern-based =========================
# BiLSTM
CUDA_VISIBLE_DEVICES=1 python main.py --dataset Weibo \
--use_preference_map False --use_pattern_based_model True --use_fact_based_model False \
--pattern_based_model 'BiLSTM' \
--lr 1e-4 --batch_size 4 --epochs 50 \
--save 'ckpts/BiLSTM'

# BiLSTM w/ Pref-FENDs
CUDA_VISIBLE_DEVICES=2 python main.py --dataset Weibo \
--use_preference_map True --use_pattern_based_model True --use_fact_based_model False \
--pattern_based_model 'BiLSTM' \
--lr 1e-4 --batch_size 4 --epochs 50 \
--save 'ckpts/BiLSTM_with_Pref-FENDs'

# EANN_Text
CUDA_VISIBLE_DEVICES=3 python main.py --dataset Weibo \
--use_preference_map False --use_pattern_based_model True --use_fact_based_model False \
--pattern_based_model 'EANN_Text' --eann_weight_of_event_loss 0.1 \
--lr 5e-5 --batch_size 32 --epochs 50 \
--save 'ckpts/EANN_Text'

# ========================= Fact-based =========================
# DeClarE
CUDA_VISIBLE_DEVICES=3 python main.py --dataset Weibo \
--use_preference_map False --use_pattern_based_model False --use_fact_based_model True \
--fact_based_model 'DeClarE' \
--lr 1e-3 --batch_size 8 --epochs 50 \
--save 'ckpts/DeClarE'

# DeClarE w/ Pref-FENDs
CUDA_VISIBLE_DEVICES=2 python main.py --dataset Weibo \
--use_preference_map True --use_pattern_based_model False --use_fact_based_model True \
--fact_based_model 'DeClarE' \
--lr 1e-3 --batch_size 8 --epochs 50 \
--save 'ckpts/DeClarE_with_Pref-FENDs'

# ========================= Integrated Models =========================
