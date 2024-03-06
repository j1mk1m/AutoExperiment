cd code
cd classification
mv model_elec.py model.py
python train.py --dataset Elec2 --learning_rate 5e-5 --num_rnn_layer 10 --hidden_dim 128